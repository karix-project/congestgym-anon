"""
carla_gym/core/dataset_adapters.py

TraceTrafficGenerator needs a CSV with columns (step, src, dst, volume),
where src/dst are integer node ids in [0, num_nodes). Real-world datasets
never arrive in that shape, so this module converts two raw formats into
it. The raw files named below are the ones the committed traces in data/
were made from; they are NOT distributed with the repository (the capture
contains real IP addresses and the flow-count file has no stated licence).
The derived traces contain node ids only.

  * cs448b_ipasn.csv   -- Stanford CS448B dataset: one row per
                          (day, local-host, remote-ASN) with a flow count.
                          No "step/src/dst/volume" at all -- it's daily
                          aggregated local->ASN flow counts.
  * output1.csv        -- a raw packet-capture export: per-packet
                          Timestamp (unix epoch), Source IP, Destination
                          IP, Protocol, Length. Some rows (ARP/broadcast)
                          have no IP and are dropped.
  * Midterm_53_group.csv -- the SAME capture as output1.csv, exported by
                          Wireshark's text UI instead: Time is relative
                          seconds, Source/Destination mix IPs, MAC-vendor
                          strings (e.g. "Vendor_00:00:00"), and the literal
                          string "Broadcast". Only the IP<->IP rows are
                          usable (they are the same ~393k rows output1.csv
                          has; the rest have no routable destination).

Design choices worth knowing about, not just the code:

  * IP/ASN identifiers are mapped to node ids by frequency rank, not a
    raw hash. The capture is dominated by one local host (appears in
    ~70% of rows); giving the single most frequent addresses their own
    dedicated node ids and hash-bucketing the long tail keeps that
    realistic imbalance instead of smearing it across every node equally.
  * Rows that reduce to src == dst after bucketing are dropped -- the
    simulator routes src->dst and has nothing to do with a self-loop.
  * The pcap adapter bins packets into fixed-width time windows
    (`bin_seconds`) to get a `step` column; the cs448b adapter uses the
    existing daily granularity as `step` directly, since it has none
    finer than that.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

_IPV4_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


def _stable_hash(value) -> int:
    """Deterministic replacement for Python's builtin `hash()`.

    `hash()` on str/bytes is randomized per-process (PYTHONHASHSEED is
    unset by default, for hash-flooding security reasons) -- using it here
    would mean every re-run of prepare_trace.py bucket-assigns the same IP
    or ASN to a *different* node id, silently producing a different
    synthetic network from the same raw CSV each time. hashlib is stable
    across processes, machines, and Python versions."""
    return int(hashlib.md5(str(value).encode("utf-8")).hexdigest(), 16)


def _topology_degree_order(topology: str, num_nodes: int,
                           seed: int | None = None) -> list[int]:
    """Real node ids of `topology`, sorted best-connected first. Used to
    decide WHERE a heavy traffic concentration should land -- a hub stuck
    on a 2-link grid corner is a bottleneck no amount of bandwidth budget
    can fix, so placement has to be topology-aware, not arbitrary.

    `seed` only matters for topology="random": pass the seed the env will be
    built with, so placement is computed on the same graph."""
    from carla_gym.core.network import NetworkModel
    net = NetworkModel(topology=topology, num_nodes=num_nodes, seed=seed)
    degrees = dict(net.graph.degree())
    return sorted(degrees, key=lambda n: (-degrees[n], n))


def _place_by_volume(logical_src: pd.Series, logical_dst: pd.Series,
                     volume: pd.Series, topology: str, num_nodes: int,
                     topology_seed: int | None = None) -> tuple[pd.Series, pd.Series]:
    """`logical_src`/`logical_dst` are provisional node ids (e.g. straight
    from a hash bucket) that carry no information about which topology
    position they'll land on. This computes each logical id's total
    traffic volume (as src + as dst) and remaps the heaviest logical ids
    onto the best-connected real node ids, so a traffic hub lands on a
    hub position instead of wherever a hash happened to put it."""
    load = (pd.Series(volume.values, index=logical_src.values)
           .groupby(level=0).sum()
           .add(pd.Series(volume.values, index=logical_dst.values).groupby(level=0).sum(),
                fill_value=0.0))
    for lid in range(num_nodes):
        if lid not in load.index:
            load.loc[lid] = 0.0
    ranked_logical = list(load.sort_values(ascending=False).index)
    degree_order = _topology_degree_order(topology, num_nodes, topology_seed)
    remap = dict(zip(ranked_logical, degree_order))
    return logical_src.map(remap), logical_dst.map(remap)


# --------------------------------------------------------------------- #
# cs448b_ipasn.csv
# --------------------------------------------------------------------- #
def convert_cs448b(path: str, out_path: str | None = None,
                    internal_nodes: int = 10, external_buckets: int = 6,
                    topology: str = "grid",
                    topology_seed: int | None = None) -> pd.DataFrame:
    """
    Local hosts (l_ipn, already 0..9) and remote ASNs (2000+, hash-bucketed
    into `external_buckets` logical slots) are jointly placed onto real
    topology node ids by total traffic volume -- the single busiest l_ipn
    or ASN bucket in the whole 92-day trace lands on the best-connected
    node, not wherever its original id or hash happened to fall. volume =
    daily flow count `f`. `step` = day index (date rank).

    Use num_nodes = internal_nodes + external_buckets with this trace.
    """
    df = pd.read_csv(path)
    need = {"date", "l_ipn", "r_asn", "f"}
    if not need <= set(df.columns):
        raise ValueError(f"cs448b adapter needs columns {need}, got {set(df.columns)}")

    if df["l_ipn"].max() >= internal_nodes:
        raise ValueError(
            f"l_ipn goes up to {df['l_ipn'].max()}, which doesn't fit in "
            f"internal_nodes={internal_nodes}"
        )

    num_nodes = internal_nodes + external_buckets
    dates = sorted(df["date"].unique())
    date_to_step = {d: i for i, d in enumerate(dates)}

    logical_src = df["l_ipn"].astype(int)
    logical_dst = internal_nodes + (df["r_asn"].apply(_stable_hash) % external_buckets)
    volume = df["f"].astype(float)
    src_nodes, dst_nodes = _place_by_volume(logical_src, logical_dst, volume,
                                            topology, num_nodes, topology_seed)

    out = pd.DataFrame({
        "step": df["date"].map(date_to_step).values,
        "src": src_nodes.values,
        "dst": dst_nodes.values,
        "volume": volume.values,
    })
    out = out[out.src != out.dst]
    out = out.groupby(["step", "src", "dst"], as_index=False)["volume"].sum()

    if out_path:
        out.to_csv(out_path, index=False)
    out.attrs["num_nodes"] = num_nodes
    out.attrs["source"] = "cs448b_ipasn"
    return out


# --------------------------------------------------------------------- #
# output1.csv / Midterm_53_group.csv (same capture, two export formats)
# --------------------------------------------------------------------- #
def convert_packet_capture(path: str, out_path: str | None = None,
                           num_nodes: int = 16, bin_seconds: float = 5.0,
                           topology: str = "grid",
                           topology_seed: int | None = None) -> pd.DataFrame:
    """
    Auto-detects whether `path` is the numeric export (Timestamp/Source
    IP/Destination IP/Length) or the Wireshark text export
    (Time/Source/Destination/Length, with non-IP endpoints like MAC
    vendor strings or "Broadcast"). Both reduce to the same packet-level
    trace once non-IP rows are dropped and time is made relative.

    Packets are binned into `bin_seconds`-wide windows -> `step`. IPs are
    hash-bucketed into logical node ids, then `_place_by_volume` remaps
    the busiest logical buckets onto the best-connected real node ids --
    this capture is dominated by one local host appearing in ~70% of
    rows, and without topology-aware placement it will land on an
    arbitrary node (including, by chance, a 2-link grid corner, which
    throttles throughput regardless of bandwidth policy). volume =
    packet Length in bytes, summed per (step, src, dst).
    """
    df = pd.read_csv(path)
    cols = set(df.columns)

    if {"Timestamp", "Source IP", "Destination IP", "Length"} <= cols:
        ts_col, src_col, dst_col = "Timestamp", "Source IP", "Destination IP"
        fmt = "pcap-numeric"
    elif {"Time", "Source", "Destination", "Length"} <= cols:
        ts_col, src_col, dst_col = "Time", "Source", "Destination"
        fmt = "pcap-wireshark-text"
    else:
        raise ValueError(
            f"Unrecognized packet-capture schema: {cols}. Expected either "
            "{Timestamp,'Source IP','Destination IP',Length} or "
            "{Time,Source,Destination,Length}."
        )

    df = df.dropna(subset=[src_col, dst_col, "Length"]).copy()
    is_ip = df[src_col].astype(str).str.match(_IPV4_RE) & \
            df[dst_col].astype(str).str.match(_IPV4_RE)
    dropped_non_ip = int((~is_ip).sum())
    df = df[is_ip]

    t = df[ts_col].astype(float)
    t = t - t.min()  # relative seconds regardless of export format
    step = (t // bin_seconds).astype(int)

    logical_src = df[src_col].apply(lambda ip: _stable_hash(ip) % num_nodes)
    logical_dst = df[dst_col].apply(lambda ip: _stable_hash(ip) % num_nodes)
    volume = df["Length"].astype(float)
    src_nodes, dst_nodes = _place_by_volume(logical_src, logical_dst, volume,
                                            topology, num_nodes, topology_seed)

    out = pd.DataFrame({
        "step": step.values,
        "src": src_nodes.values,
        "dst": dst_nodes.values,
        "volume": volume.values,
    })
    out = out[out.src != out.dst]
    out = out.groupby(["step", "src", "dst"], as_index=False)["volume"].sum()

    if out_path:
        out.to_csv(out_path, index=False)
    out.attrs["num_nodes"] = num_nodes
    out.attrs["source"] = fmt
    out.attrs["dropped_non_ip_rows"] = dropped_non_ip
    return out


ADAPTERS = {
    "cs448b": convert_cs448b,
    "pcap": convert_packet_capture,
}


def detect_format(path: str) -> str:
    cols = set(pd.read_csv(path, nrows=5).columns)
    if {"date", "l_ipn", "r_asn", "f"} <= cols:
        return "cs448b"
    if {"Timestamp", "Source IP", "Destination IP", "Length"} <= cols or \
       {"Time", "Source", "Destination", "Length"} <= cols:
        return "pcap"
    raise ValueError(f"Could not auto-detect format from columns: {cols}")
