"""
pcap_extractor.py
Robust Bi-directional Flow Extractor for PCAP / PCAPNG Network Captures
Converts raw packet captures into standard 78-feature CICFlowMeter feature vectors.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import scapy.all as scapy
except ImportError:
    scapy = None


def extract_flows_from_pcap(pcap_path_or_bytes, feature_names=None):
    """
    Parse a PCAP / PCAPNG file (path or BytesIO) and reconstruct bidirectional flows.
    Returns: (flow_dataframe, packet_metadata_list)
    """
    if scapy is None:
        raise RuntimeError("Scapy is not available in the active Python environment.")

    # Load packets
    if isinstance(pcap_path_or_bytes, str):
        packets = scapy.rdpcap(pcap_path_or_bytes)
    else:
        # Save temp file if bytes
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
            tmp.write(pcap_path_or_bytes.read() if hasattr(pcap_path_or_bytes, 'read') else pcap_path_or_bytes)
            tmp_path = tmp.name
        try:
            packets = scapy.rdpcap(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if len(packets) == 0:
        return pd.DataFrame(), []

    # Map packets into bi-directional flows: key = (min(ip1, ip2), max(ip1, ip2), min(p1, p2), max(p1, p2), proto)
    flows = defaultdict(lambda: {
        'forward_src': None,
        'dst_port': 0,
        'proto': 6,
        'fwd_pkts': [],
        'bwd_pkts': [],
        'fwd_times': [],
        'bwd_times': [],
        'fwd_lens': [],
        'bwd_lens': [],
        'fwd_flags': [],
        'bwd_flags': [],
        'fwd_win': [],
        'bwd_win': [],
        'fwd_hdr_len': 0,
        'bwd_hdr_len': 0
    })

    packet_meta = []

    for pkt in packets:
        if not pkt.haslayer(scapy.IP):
            continue

        ip_layer = pkt[scapy.IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = ip_layer.proto
        pkt_time = float(pkt.time)
        pkt_len = len(pkt)

        sport = 0
        dport = 0
        tcp_flags = 0
        win_size = 0
        hdr_len = 20

        if pkt.haslayer(scapy.TCP):
            tcp = pkt[scapy.TCP]
            sport = tcp.sport
            dport = tcp.dport
            tcp_flags = int(tcp.flags)
            win_size = tcp.window
            hdr_len = tcp.dataofs * 4 if hasattr(tcp, 'dataofs') and tcp.dataofs else 20
        elif pkt.haslayer(scapy.UDP):
            udp = pkt[scapy.UDP]
            sport = udp.sport
            dport = udp.dport
            hdr_len = 8

        # Flow 5-tuple direction key
        flow_endpoints = (min(src_ip, dst_ip), max(src_ip, dst_ip), min(sport, dport), max(sport, dport), proto)
        flow_data = flows[flow_endpoints]

        # First packet establishes forward direction
        if flow_data['forward_src'] is None:
            flow_data['forward_src'] = src_ip
            flow_data['dst_port'] = dport
            flow_data['proto'] = proto

        is_forward = (src_ip == flow_data['forward_src'])

        if is_forward:
            flow_data['fwd_pkts'].append(pkt)
            flow_data['fwd_times'].append(pkt_time)
            flow_data['fwd_lens'].append(pkt_len)
            flow_data['fwd_flags'].append(tcp_flags)
            flow_data['fwd_win'].append(win_size)
            flow_data['fwd_hdr_len'] += hdr_len
        else:
            flow_data['bwd_pkts'].append(pkt)
            flow_data['bwd_times'].append(pkt_time)
            flow_data['bwd_lens'].append(pkt_len)
            flow_data['bwd_flags'].append(tcp_flags)
            flow_data['bwd_win'].append(win_size)
            flow_data['bwd_hdr_len'] += hdr_len

        packet_meta.append({
            'timestamp': pkt_time,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': sport,
            'dst_port': dport,
            'protocol': 'TCP' if proto == 6 else ('UDP' if proto == 17 else str(proto)),
            'length': pkt_len
        })

    # Convert flows into 78-feature dictionary records
    flow_rows = []

    for endpoints, fd in flows.items():
        all_times = sorted(fd['fwd_times'] + fd['bwd_times'])
        if len(all_times) == 0:
            continue

        flow_duration_us = (all_times[-1] - all_times[0]) * 1e6
        if flow_duration_us <= 0:
            flow_duration_us = 1.0

        fwd_lens = fd['fwd_lens']
        bwd_lens = fd['bwd_lens']
        all_lens = fwd_lens + bwd_lens

        tot_fwd_pkts = len(fwd_lens)
        tot_bwd_pkts = len(bwd_lens)
        tot_len_fwd = sum(fwd_lens)
        tot_len_bwd = sum(bwd_lens)
        tot_bytes = tot_len_fwd + tot_len_bwd
        tot_pkts = tot_fwd_pkts + tot_bwd_pkts

        # IAT calculations
        def get_iats(times):
            if len(times) < 2:
                return [0.0]
            s = sorted(times)
            return [(s[i] - s[i-1]) * 1e6 for i in range(1, len(s))]

        flow_iats = get_iats(all_times)
        fwd_iats = get_iats(fd['fwd_times'])
        bwd_iats = get_iats(fd['bwd_times'])

        flow_sec = flow_duration_us / 1e6

        # TCP Flag parsing
        all_flags = fd['fwd_flags'] + fd['bwd_flags']
        fin_cnt = sum(1 for fl in all_flags if fl & 0x01)
        syn_cnt = sum(1 for fl in all_flags if fl & 0x02)
        rst_cnt = sum(1 for fl in all_flags if fl & 0x04)
        psh_cnt = sum(1 for fl in all_flags if fl & 0x08)
        ack_cnt = sum(1 for fl in all_flags if fl & 0x10)
        urg_cnt = sum(1 for fl in all_flags if fl & 0x20)
        ece_cnt = sum(1 for fl in all_flags if fl & 0x40)
        cwe_cnt = sum(1 for fl in all_flags if fl & 0x80)

        fwd_psh = sum(1 for fl in fd['fwd_flags'] if fl & 0x08)
        bwd_psh = sum(1 for fl in fd['bwd_flags'] if fl & 0x08)
        fwd_urg = sum(1 for fl in fd['fwd_flags'] if fl & 0x20)
        bwd_urg = sum(1 for fl in fd['bwd_flags'] if fl & 0x20)

        init_win_fwd = fd['fwd_win'][0] if fd['fwd_win'] else 0
        init_win_bwd = fd['bwd_win'][0] if fd['bwd_win'] else 0

        # Build feature map
        row = {
            'Destination Port': fd['dst_port'],
            'Flow Duration': flow_duration_us,
            'Total Fwd Packets': tot_fwd_pkts,
            'Total Backward Packets': tot_bwd_pkts,
            'Total Length of Fwd Packets': tot_len_fwd,
            'Total Length of Bwd Packets': tot_len_bwd,
            'Fwd Packet Length Max': max(fwd_lens) if fwd_lens else 0,
            'Fwd Packet Length Min': min(fwd_lens) if fwd_lens else 0,
            'Fwd Packet Length Mean': np.mean(fwd_lens) if fwd_lens else 0,
            'Fwd Packet Length Std': np.std(fwd_lens) if len(fwd_lens) > 1 else 0,
            'Bwd Packet Length Max': max(bwd_lens) if bwd_lens else 0,
            'Bwd Packet Length Min': min(bwd_lens) if bwd_lens else 0,
            'Bwd Packet Length Mean': np.mean(bwd_lens) if bwd_lens else 0,
            'Bwd Packet Length Std': np.std(bwd_lens) if len(bwd_lens) > 1 else 0,
            'Flow Bytes/s': tot_bytes / flow_sec if flow_sec > 0 else 0,
            'Flow Packets/s': tot_pkts / flow_sec if flow_sec > 0 else 0,
            'Flow IAT Mean': np.mean(flow_iats),
            'Flow IAT Std': np.std(flow_iats) if len(flow_iats) > 1 else 0,
            'Flow IAT Max': max(flow_iats),
            'Flow IAT Min': min(flow_iats),
            'Fwd IAT Total': sum(fwd_iats),
            'Fwd IAT Mean': np.mean(fwd_iats),
            'Fwd IAT Std': np.std(fwd_iats) if len(fwd_iats) > 1 else 0,
            'Fwd IAT Max': max(fwd_iats),
            'Fwd IAT Min': min(fwd_iats),
            'Bwd IAT Total': sum(bwd_iats),
            'Bwd IAT Mean': np.mean(bwd_iats),
            'Bwd IAT Std': np.std(bwd_iats) if len(bwd_iats) > 1 else 0,
            'Bwd IAT Max': max(bwd_iats),
            'Bwd IAT Min': min(bwd_iats),
            'Fwd PSH Flags': fwd_psh,
            'Bwd PSH Flags': bwd_psh,
            'Fwd URG Flags': fwd_urg,
            'Bwd URG Flags': bwd_urg,
            'Fwd Header Length': fd['fwd_hdr_len'],
            'Bwd Header Length': fd['bwd_hdr_len'],
            'Fwd Packets/s': tot_fwd_pkts / flow_sec if flow_sec > 0 else 0,
            'Bwd Packets/s': tot_bwd_pkts / flow_sec if flow_sec > 0 else 0,
            'Min Packet Length': min(all_lens) if all_lens else 0,
            'Max Packet Length': max(all_lens) if all_lens else 0,
            'Packet Length Mean': np.mean(all_lens) if all_lens else 0,
            'Packet Length Std': np.std(all_lens) if len(all_lens) > 1 else 0,
            'Packet Length Variance': np.var(all_lens) if len(all_lens) > 1 else 0,
            'FIN Flag Count': fin_cnt,
            'SYN Flag Count': syn_cnt,
            'RST Flag Count': rst_cnt,
            'PSH Flag Count': psh_cnt,
            'ACK Flag Count': ack_cnt,
            'URG Flag Count': urg_cnt,
            'CWE Flag Count': cwe_cnt,
            'ECE Flag Count': ece_cnt,
            'Down/Up Ratio': (tot_bwd_pkts / tot_fwd_pkts) if tot_fwd_pkts > 0 else 0,
            'Average Packet Size': np.mean(all_lens) if all_lens else 0,
            'Avg Fwd Segment Size': np.mean(fwd_lens) if fwd_lens else 0,
            'Avg Bwd Segment Size': np.mean(bwd_lens) if bwd_lens else 0,
            'Fwd Header Length.1': fd['fwd_hdr_len'],
            'Fwd Avg Bytes/Bulk': 0,
            'Fwd Avg Packets/Bulk': 0,
            'Fwd Avg Bulk Rate': 0,
            'Bwd Avg Bytes/Bulk': 0,
            'Bwd Avg Packets/Bulk': 0,
            'Bwd Avg Bulk Rate': 0,
            'Subflow Fwd Packets': tot_fwd_pkts,
            'Subflow Fwd Bytes': tot_len_fwd,
            'Subflow Bwd Packets': tot_bwd_pkts,
            'Subflow Bwd Bytes': tot_len_bwd,
            'Init_Win_bytes_forward': init_win_fwd,
            'Init_Win_bytes_backward': init_win_bwd,
            'act_data_pkt_fwd': max(0, tot_fwd_pkts - 1),
            'min_seg_size_forward': 20,
            'Active Mean': 0,
            'Active Std': 0,
            'Active Max': 0,
            'Active Min': 0,
            'Idle Mean': 0,
            'Idle Std': 0,
            'Idle Max': 0,
            'Idle Min': 0,
            '_SRC_IP': endpoints[0],
            '_DST_IP': endpoints[1],
            '_PROTO': 'TCP' if fd['proto'] == 6 else ('UDP' if fd['proto'] == 17 else str(fd['proto']))
        }

        flow_rows.append(row)

    df_out = pd.DataFrame(flow_rows)

    # Ensure all required 78 features are present
    if feature_names:
        for fn in feature_names:
            if fn not in df_out.columns:
                df_out[fn] = 0

    return df_out, packet_meta
