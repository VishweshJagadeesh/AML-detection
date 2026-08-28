"""
Graph-Based Anti-Money-Laundering (AML) Analytics Platform
============================================================

Traditional transaction monitoring scores one transaction at a time:

    Transaction -> transaction-level features -> ML / rules -> alert

This project instead represents the whole ledger as an account graph and
scores *accounts* using the shape of their neighbourhood in that graph:

    Transactions -> Account Graph -> Graph Features -> ML / Risk Model
                 -> Suspicious Accounts -> Investigation Network

Money laundering is fundamentally a *relational* crime (funds are moved
between accounts to obscure origin), so structural signals - cycles,
fan-out/fan-in, centrality, community structure - carry information that a
single transaction's amount and timestamp cannot.

Pipeline (see `main()`):
    1. TransactionGenerator  - synthesize a transaction ledger with realistic
       "normal" behaviour and deliberately injected laundering topologies.
    2. TransactionGraph      - build a directed account graph from the ledger.
    3. GraphFeatureEngineer  - compute node-level graph + transactional
       features.
    4. AMLScorer             - compare a rule-based scorer, Logistic
       Regression, and Random Forest at predicting the (pattern-based)
       ground-truth suspicious label.
    5. InvestigationEngine   - produce a human-readable investigation packet
       for a flagged account.

Run with:  python aml_graph.py

IMPORTANT: All transaction data is synthetically generated for this
portfolio project. Nothing in this file implies regulatory compliance,
KYC/CDD coverage, or sanctions screening - see README.md, section
"Limitations", for what a production AML system additionally requires.
"""

from __future__ import annotations

import random
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CURRENCY_SYMBOL = "\u20b9"  # INR symbol; amounts are synthetic, currency is illustrative


# ---------------------------------------------------------------------------
# 1. TRANSACTION GENERATOR
# ---------------------------------------------------------------------------
class TransactionGenerator:
    """
    Generates a synthetic transaction ledger.

    Suspicious labels are NEVER assigned at random. An account is only
    labeled suspicious if it was deliberately placed inside one of the
    laundering topologies below (circular flows, fan-out, fan-in, rapid
    pass-through, structuring, unusual cross-border transfers). The label is
    used solely to evaluate, after the fact, whether graph-derived features
    can recover these patterns - it is never used as a feature itself.
    """

    COUNTRIES = ["IN", "US", "UK", "AE", "SG", "DE", "CH", "KY", "BM", "PA"]
    COUNTRY_WEIGHTS = [40, 15, 10, 8, 8, 6, 5, 3, 3, 2]
    HIGH_RISK_COUNTRIES = {"KY", "BM", "PA"}  # classic offshore / shell jurisdictions
    TXN_TYPES = ["WIRE", "CARD", "UPI", "ACH", "CASH_DEPOSIT"]

    def __init__(
        self,
        n_normal_accounts: int = 250,
        n_normal_txns: int = 3000,
        start_date: datetime = datetime(2024, 1, 1),
        seed: int = RANDOM_SEED,
    ):
        self.n_normal_accounts = n_normal_accounts
        self.n_normal_txns = n_normal_txns
        self.start_date = start_date
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.accounts: List[str] = [f"ACC{i:04d}" for i in range(n_normal_accounts)]
        self.transactions: List[dict] = []
        self.suspicious_accounts: Set[str] = set()
        self.pattern_membership: Dict[str, Set[str]] = defaultdict(set)
        self._txn_counter = 0

    # -- helpers -------------------------------------------------------
    def _new_txn_id(self) -> str:
        self._txn_counter += 1
        return f"TXN{self._txn_counter:06d}"

    def _rand_time(self, day_span: float = 180) -> datetime:
        return self.start_date + timedelta(days=self.rng.uniform(0, day_span))

    def _add_txn(self, src, dst, amount, ts, txn_type, country, pattern="normal"):
        self.transactions.append(
            {
                "txn_id": self._new_txn_id(),
                "account_id": src,
                "counterparty_id": dst,
                "amount": round(float(amount), 2),
                "timestamp": ts,
                "transaction_type": txn_type,
                "country": country,
                "pattern": pattern,
            }
        )

    # -- normal behaviour -----------------------------------------------
    def generate_normal_transactions(self):
        """Ordinary peer-to-peer activity: log-normal amounts, mixed
        transaction types, mostly domestic-weighted countries."""
        for _ in range(self.n_normal_txns):
            src, dst = self.rng.sample(self.accounts, 2)
            amount = max(10.0, float(self.np_rng.lognormal(mean=6.5, sigma=1.0)))
            ts = self._rand_time()
            txn_type = self.rng.choice(self.TXN_TYPES)
            country = self.rng.choices(self.COUNTRIES, weights=self.COUNTRY_WEIGHTS)[0]
            self._add_txn(src, dst, amount, ts, txn_type, country)

    # -- injected laundering topologies -----------------------------------
    def inject_circular_flows(self, n_cycles: int = 6, cycle_len: int = 3):
        """A -> B -> C -> A: funds return (minus a small skim) to origin."""
        for i in range(n_cycles):
            ring = [f"RING{i}_{j}" for j in range(cycle_len)]
            self.accounts.extend(ring)
            base_amount = self.rng.uniform(8000, 40000)
            base_ts = self._rand_time()
            for j in range(cycle_len):
                src, dst = ring[j], ring[(j + 1) % cycle_len]
                amount = base_amount * self.rng.uniform(0.92, 0.99)
                ts = base_ts + timedelta(minutes=15 * j)
                self._add_txn(src, dst, amount, ts, "WIRE", "IN", pattern="circular")
                self.pattern_membership["circular"].update([src, dst])
            self.suspicious_accounts.update(ring)

    def inject_fan_out(self, n_hubs: int = 5, spokes: int = 6):
        """One account rapidly disperses funds to many distinct accounts."""
        for i in range(n_hubs):
            hub = f"FANOUT{i}"
            self.accounts.append(hub)
            base_ts = self._rand_time()
            for k in range(spokes):
                spoke = f"FANOUT{i}_S{k}"
                self.accounts.append(spoke)
                amount = self.rng.uniform(2000, 9000)
                ts = base_ts + timedelta(minutes=self.rng.uniform(0, 30))
                self._add_txn(hub, spoke, amount, ts, "ACH", "IN", pattern="fan_out")
                self.pattern_membership["fan_out"].update([hub, spoke])
            self.suspicious_accounts.add(hub)

    def inject_fan_in(self, n_hubs: int = 5, spokes: int = 6):
        """Many distinct accounts rapidly funnel funds into one account."""
        for i in range(n_hubs):
            hub = f"FANIN{i}"
            self.accounts.append(hub)
            base_ts = self._rand_time()
            for k in range(spokes):
                spoke = f"FANIN{i}_S{k}"
                self.accounts.append(spoke)
                amount = self.rng.uniform(2000, 9000)
                ts = base_ts + timedelta(minutes=self.rng.uniform(0, 30))
                self._add_txn(spoke, hub, amount, ts, "ACH", "IN", pattern="fan_in")
                self.pattern_membership["fan_in"].update([hub, spoke])
            self.suspicious_accounts.add(hub)

    def inject_rapid_passthrough(self, n_chains: int = 6):
        """A -> B -> C within minutes, near-identical amount (mule chain)."""
        for i in range(n_chains):
            a, b, c = f"PASS{i}_A", f"PASS{i}_B", f"PASS{i}_C"
            self.accounts.extend([a, b, c])
            amount = self.rng.uniform(15000, 60000)
            base_ts = self._rand_time()
            self._add_txn(a, b, amount, base_ts, "WIRE", "SG", pattern="passthrough")
            self._add_txn(
                b,
                c,
                amount * self.rng.uniform(0.97, 0.995),
                base_ts + timedelta(minutes=self.rng.uniform(2, 8)),
                "WIRE",
                "SG",
                pattern="passthrough",
            )
            self.pattern_membership["passthrough"].update([a, b, c])
            self.suspicious_accounts.update([a, b, c])

    def inject_structuring(self, n_structurers: int = 6, threshold: float = 10000):
        """Multiple transactions clustered just under a reporting threshold."""
        for i in range(n_structurers):
            acc = f"STRUCT{i}"
            self.accounts.append(acc)
            base_ts = self._rand_time()
            n_txns = self.rng.randint(6, 10)
            for _ in range(n_txns):
                amount = threshold * self.rng.uniform(0.90, 0.99)
                ts = base_ts + timedelta(hours=self.rng.uniform(0, 72))
                dst = self.rng.choice(self.accounts[: self.n_normal_accounts])
                self._add_txn(acc, dst, amount, ts, "CASH_DEPOSIT", "IN", pattern="structuring")
            self.pattern_membership["structuring"].add(acc)
            self.suspicious_accounts.add(acc)

    def inject_unusual_cross_border(self, n_accounts: int = 6):
        """Large, frequent transfers to offshore/shell-heavy jurisdictions."""
        for i in range(n_accounts):
            acc = f"XBORDER{i}"
            self.accounts.append(acc)
            base_ts = self._rand_time()
            for _ in range(self.rng.randint(3, 5)):
                dst = f"{acc}_DST{self.rng.randint(0, 2)}"
                self.accounts.append(dst)
                amount = self.rng.uniform(20000, 80000)
                ts = base_ts + timedelta(hours=self.rng.uniform(0, 48))
                country = self.rng.choice(list(self.HIGH_RISK_COUNTRIES))
                self._add_txn(acc, dst, amount, ts, "WIRE", country, pattern="cross_border")
                self.pattern_membership["cross_border"].update([acc, dst])
            self.suspicious_accounts.add(acc)

    def generate(self) -> pd.DataFrame:
        self.generate_normal_transactions()
        self.inject_circular_flows()
        self.inject_fan_out()
        self.inject_fan_in()
        self.inject_rapid_passthrough()
        self.inject_structuring()
        self.inject_unusual_cross_border()
        self.accounts = list(dict.fromkeys(self.accounts))  # de-dup, keep order
        df = pd.DataFrame(self.transactions)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


# ---------------------------------------------------------------------------
# 2. TRANSACTION GRAPH
# ---------------------------------------------------------------------------
class TransactionGraph:
    """
    Wraps the ledger as a directed multigraph (node = account, edge =
    transaction) and exposes derived views used by feature engineering.

    NetworkX is used here because:
      - it ships pure-Python implementations of every algorithm this project
        needs (PageRank, degree/betweenness centrality, community detection,
        connected components) with no extra infrastructure;
      - it consumes pandas edge lists directly, keeping the
        "transactions -> graph" step a few lines of code;
      - at portfolio scale (hundreds to low thousands of nodes) its
        pure-Python performance is entirely adequate and runs identically on
        any machine, satisfying the "runs locally" constraint without a
        GPU-graph dependency.
    """

    def __init__(self, transactions: pd.DataFrame):
        self.df = transactions
        self.graph = nx.MultiDiGraph()
        self._build()

    def _build(self):
        for _, row in self.df.iterrows():
            self.graph.add_edge(
                row["account_id"],
                row["counterparty_id"],
                amount=row["amount"],
                timestamp=row["timestamp"],
                country=row["country"],
                transaction_type=row["transaction_type"],
                txn_id=row["txn_id"],
            )

    def simple_graph(self) -> nx.DiGraph:
        """Collapsed simple directed graph: one edge per (src, dst) pair,
        with amount summed and transaction count kept."""
        g = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            if g.has_edge(u, v):
                g[u][v]["amount"] += data["amount"]
                g[u][v]["count"] += 1
            else:
                g.add_edge(u, v, amount=data["amount"], count=1)
        return g

    def connected_components(self) -> List[Set[str]]:
        return list(nx.weakly_connected_components(self.graph))

    def short_cycles(self, max_len: int = 4) -> Tuple[List[Tuple[str, ...]], Dict[str, int]]:
        """
        Bounded-depth DFS cycle search on the collapsed simple graph.

        A full `nx.simple_cycles()` over the raw multigraph is combinatorially
        expensive and not what AML typology needs anyway: laundering rings
        are deliberately kept short (2-5 hops) because every extra hop dilutes
        the funds and adds detection surface for the launderer. We therefore
        only enumerate simple cycles up to `max_len`, which both bounds
        runtime and matches real ring sizes.

        Returns (list_of_cycles, node -> number_of_cycles_it_participates_in).
        """
        g = self.simple_graph()
        found: List[Tuple[str, ...]] = []
        seen_keys = set()

        for start in g.nodes:
            stack = [(start, (start,))]
            while stack:
                current, path = stack.pop()
                if len(path) > max_len:
                    continue
                for nxt in g.successors(current):
                    if nxt == start and len(path) >= 2:
                        key = (frozenset(path), len(path))
                        if key not in seen_keys:
                            seen_keys.add(key)
                            found.append(path)
                    elif nxt not in path and len(path) < max_len:
                        stack.append((nxt, path + (nxt,)))

        node_cycle_count: Dict[str, int] = defaultdict(int)
        for cyc in found:
            for n in cyc:
                node_cycle_count[n] += 1
        return found, node_cycle_count


# ---------------------------------------------------------------------------
# 3. GRAPH FEATURE ENGINEER
# ---------------------------------------------------------------------------
class GraphFeatureEngineer:
    """
    Turns the transaction graph into a node-level feature table - the
    "Account Graph -> Graph Features" step. Every feature here is either a
    graph-structural property (degree, centrality, community, cycles) or a
    ledger aggregate (amounts, velocity, country mix); none of them use the
    ground-truth label.
    """

    def __init__(self, tx_graph: TransactionGraph, structuring_threshold: float = 10000):
        self.tx_graph = tx_graph
        self.g = tx_graph.simple_graph()
        self.threshold = structuring_threshold
        self.df = tx_graph.df

    def build_features(self) -> pd.DataFrame:
        g = self.g
        nodes = list(g.nodes)

        in_deg = dict(g.in_degree())
        out_deg = dict(g.out_degree())

        pagerank = nx.pagerank(g, alpha=0.85) if g.number_of_edges() > 0 else {n: 0.0 for n in nodes}
        degree_centrality = nx.degree_centrality(g)

        # Betweenness is O(V*E); approximate via sampling on larger graphs.
        if g.number_of_nodes() > 400:
            k = min(200, g.number_of_nodes())
            betweenness = nx.betweenness_centrality(g, k=k, seed=RANDOM_SEED)
        else:
            betweenness = nx.betweenness_centrality(g)

        undirected = g.to_undirected()
        clustering = nx.clustering(undirected)

        communities = list(nx.algorithms.community.greedy_modularity_communities(undirected))
        community_of: Dict[str, int] = {}
        for idx, com in enumerate(communities):
            for n in com:
                community_of[n] = idx

        _, node_cycle_count = self.tx_graph.short_cycles(max_len=4)

        grouped_out = self.df.groupby("account_id")
        grouped_in = self.df.groupby("counterparty_id")

        rows = []
        for n in nodes:
            out_txns = grouped_out.get_group(n) if n in grouped_out.groups else pd.DataFrame(
                columns=self.df.columns
            )
            in_txns = grouped_in.get_group(n) if n in grouped_in.groups else pd.DataFrame(
                columns=self.df.columns
            )

            total_out = float(out_txns["amount"].sum()) if len(out_txns) else 0.0
            total_in = float(in_txns["amount"].sum()) if len(in_txns) else 0.0
            unique_out_cp = out_txns["counterparty_id"].nunique() if len(out_txns) else 0
            unique_in_cp = in_txns["account_id"].nunique() if len(in_txns) else 0

            all_txns = pd.concat([out_txns, in_txns]) if len(out_txns) or len(in_txns) else pd.DataFrame(
                columns=self.df.columns
            )
            n_txns = len(all_txns)
            if n_txns >= 2:
                span_seconds = (all_txns["timestamp"].max() - all_txns["timestamp"].min()).total_seconds()
                velocity = n_txns / max(span_seconds / 3600.0, 1e-6)
            else:
                velocity = 0.0

            if len(out_txns):
                near_threshold = out_txns[
                    (out_txns["amount"] >= self.threshold * 0.85) & (out_txns["amount"] < self.threshold)
                ]
                structuring_ratio = len(near_threshold) / len(out_txns)
            else:
                structuring_ratio = 0.0

            if len(all_txns):
                high_risk_ratio = float(
                    all_txns["country"].isin(TransactionGenerator.HIGH_RISK_COUNTRIES).mean()
                )
            else:
                high_risk_ratio = 0.0

            fan_out_ratio = unique_out_cp / max(out_deg.get(n, 0), 1)
            fan_in_ratio = unique_in_cp / max(in_deg.get(n, 0), 1)

            rows.append(
                {
                    "account_id": n,
                    "in_degree": in_deg.get(n, 0),
                    "out_degree": out_deg.get(n, 0),
                    "total_degree": in_deg.get(n, 0) + out_deg.get(n, 0),
                    "total_in_amount": total_in,
                    "total_out_amount": total_out,
                    "net_flow": total_in - total_out,
                    "unique_in_counterparties": unique_in_cp,
                    "unique_out_counterparties": unique_out_cp,
                    "fan_in_ratio": fan_in_ratio,
                    "fan_out_ratio": fan_out_ratio,
                    "txn_velocity_per_hr": velocity,
                    "pagerank": pagerank.get(n, 0.0),
                    "degree_centrality": degree_centrality.get(n, 0.0),
                    "betweenness_centrality": betweenness.get(n, 0.0),
                    "clustering_coeff": clustering.get(n, 0.0),
                    "community_id": community_of.get(n, -1),
                    "short_cycle_count": node_cycle_count.get(n, 0),
                    "structuring_ratio": structuring_ratio,
                    "high_risk_country_ratio": high_risk_ratio,
                    "n_transactions": n_txns,
                }
            )

        features_df = pd.DataFrame(rows)
        community_sizes = features_df["community_id"].value_counts().to_dict()
        features_df["community_size"] = features_df["community_id"].map(community_sizes)
        return features_df


# ---------------------------------------------------------------------------
# 4. AML SCORER
# ---------------------------------------------------------------------------
class AMLScorer:
    """
    Compares three ways to turn graph features into an account suspicion
    score:

      1. Rule-based   - fixed, hand-written weights on normalized graph
         signals. This mirrors what most legacy AML systems run today:
         transparent, cheap, but rigid.
      2. Logistic Regression - a simple, auditable linear model. Regulators
         and model-risk teams favor models whose coefficients can be
         explained account-by-account.
      3. Random Forest - a non-linear model capable of picking up feature
         interactions (e.g. high fan-out *and* a short cycle *and* near-
         threshold amounts together, which no single rule threshold catches
         cleanly).
    """

    RULE_WEIGHTS = {
        "fan_out_ratio": 0.15,
        "fan_in_ratio": 0.15,
        "short_cycle_count": 0.20,
        "structuring_ratio": 0.20,
        "high_risk_country_ratio": 0.15,
        "txn_velocity_per_hr": 0.15,
    }

    FEATURE_COLS = [
        "in_degree",
        "out_degree",
        "total_degree",
        "total_in_amount",
        "total_out_amount",
        "net_flow",
        "unique_in_counterparties",
        "unique_out_counterparties",
        "fan_in_ratio",
        "fan_out_ratio",
        "txn_velocity_per_hr",
        "pagerank",
        "degree_centrality",
        "betweenness_centrality",
        "clustering_coeff",
        "short_cycle_count",
        "structuring_ratio",
        "high_risk_country_ratio",
        "n_transactions",
        "community_size",
    ]

    def __init__(self, features_df: pd.DataFrame):
        self.features_df = features_df.copy()
        self.lr_model = None
        self.rf_model = None
        self.scaler = None
        self.y_test = None
        self.test_index = None

    def rule_based_score(self) -> pd.Series:
        df = self.features_df

        def norm(s: pd.Series) -> pd.Series:
            return (s - s.min()) / (s.max() - s.min() + 1e-9)

        score = (
            self.RULE_WEIGHTS["fan_out_ratio"] * norm(df["fan_out_ratio"])
            + self.RULE_WEIGHTS["fan_in_ratio"] * norm(df["fan_in_ratio"])
            + self.RULE_WEIGHTS["short_cycle_count"] * norm(df["short_cycle_count"])
            + self.RULE_WEIGHTS["structuring_ratio"] * norm(df["structuring_ratio"])
            + self.RULE_WEIGHTS["high_risk_country_ratio"] * norm(df["high_risk_country_ratio"])
            + self.RULE_WEIGHTS["txn_velocity_per_hr"] * norm(df["txn_velocity_per_hr"])
        )
        return score

    def train_and_evaluate(self, labels: pd.Series, test_size: float = 0.3, seed: int = RANDOM_SEED):
        df = self.features_df
        X = df[self.FEATURE_COLS].fillna(0.0)
        y = labels.values

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=test_size, random_state=seed, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        results = {}

        # --- 1. Rule-based: no training, threshold the rule score --------
        rule_scores_all = self.rule_based_score()
        rule_test_scores = rule_scores_all.loc[idx_test].values
        alert_threshold = np.quantile(rule_scores_all.values, 0.90)
        rule_pred = (rule_test_scores >= alert_threshold).astype(int)
        results["Rule-Based"] = self._evaluate(y_test, rule_pred, rule_test_scores)

        # --- 2. Logistic Regression ---------------------------------------
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        lr.fit(X_train_s, y_train)
        lr_scores = lr.predict_proba(X_test_s)[:, 1]
        lr_pred = lr.predict(X_test_s)
        results["Logistic Regression"] = self._evaluate(y_test, lr_pred, lr_scores)
        self.lr_model, self.scaler = lr, scaler

        # --- 3. Random Forest ---------------------------------------------
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced_subsample", random_state=seed
        )
        rf.fit(X_train, y_train)
        rf_scores = rf.predict_proba(X_test)[:, 1]
        rf_pred = rf.predict(X_test)
        results["Random Forest"] = self._evaluate(y_test, rf_pred, rf_scores)
        self.rf_model = rf

        importances = pd.Series(rf.feature_importances_, index=self.FEATURE_COLS).sort_values(
            ascending=False
        )

        self.test_index = idx_test
        self.y_test = y_test

        return results, importances

    @staticmethod
    def _evaluate(y_true, y_pred, scores) -> dict:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_true, scores) if len(set(y_true)) > 1 else float("nan")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        order = np.argsort(-scores)
        k = max(1, int(0.1 * len(y_true)))
        top_k_idx = order[:k]
        precision_at_k = float(np.asarray(y_true)[top_k_idx].sum() / k)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": pr_auc,
            "confusion_matrix": cm.tolist(),
            "precision_at_k": precision_at_k,
            "k": k,
            "alerts_generated": int(np.sum(y_pred)),
            "true_positives": int(cm[1, 1]),
            "false_positives": int(cm[0, 1]),
        }


# ---------------------------------------------------------------------------
# 5. INVESTIGATION ENGINE
# ---------------------------------------------------------------------------
class InvestigationEngine:
    """
    Assembles a human-readable investigation packet for a flagged account:
    risk score, direct counterparties, amounts, timestamps, suspicious
    cycles, centrality measures, and other accounts connected via those
    cycles. This is the "Suspicious Accounts -> Investigation Network" step.
    """

    def __init__(self, tx_graph: TransactionGraph, features_df: pd.DataFrame, risk_scores: pd.Series):
        self.tx_graph = tx_graph
        self.df = tx_graph.df
        self.features_df = features_df.set_index("account_id")
        self.risk_scores = risk_scores
        self.g = tx_graph.simple_graph()

    def investigate(self, account_id: str) -> str:
        if account_id not in self.features_df.index:
            return f"Account {account_id} not found in graph."

        feats = self.features_df.loc[account_id]
        risk = self.risk_scores.get(account_id, float("nan"))

        out_txns = self.df[self.df["account_id"] == account_id]
        in_txns = self.df[self.df["counterparty_id"] == account_id]
        all_txns = pd.concat([out_txns, in_txns]).sort_values("timestamp")

        direct_out = sorted(out_txns["counterparty_id"].unique().tolist())
        direct_in = sorted(in_txns["account_id"].unique().tolist())

        all_cycles, _ = self.tx_graph.short_cycles(max_len=4)
        my_cycles = [c for c in all_cycles if account_id in c]

        connected_suspicious: Set[str] = set()
        for c in my_cycles:
            connected_suspicious.update(c)
        connected_suspicious.discard(account_id)

        lines = []
        lines.append(f"Account {account_id}")
        lines.append(f"Risk Score: {risk:.2f}")
        lines.append("")
        lines.append(f"Direct outgoing counterparties ({len(direct_out)}): {direct_out[:10]}")
        lines.append(f"Direct incoming counterparties ({len(direct_in)}): {direct_in[:10]}")
        lines.append("")

        if my_cycles:
            lines.append("Suspicious relationships (short cycles found):")
            for c in my_cycles[:5]:
                lines.append("  " + " -> ".join(c) + f" -> {c[0]}")
        else:
            lines.append("Suspicious relationships: none found within a 4-hop cycle search.")
        lines.append("")

        total_flow = float(all_txns["amount"].sum()) if len(all_txns) else 0.0
        lines.append(f"Total flow through account: {CURRENCY_SYMBOL}{total_flow:,.2f}")
        lines.append(f"  Total incoming: {CURRENCY_SYMBOL}{feats['total_in_amount']:,.2f}")
        lines.append(f"  Total outgoing: {CURRENCY_SYMBOL}{feats['total_out_amount']:,.2f}")
        lines.append("")

        if len(all_txns) >= 2:
            span_minutes = (all_txns["timestamp"].max() - all_txns["timestamp"].min()).total_seconds() / 60
            lines.append(f"Velocity: {len(all_txns)} transactions within {span_minutes:.1f} minutes")
        else:
            lines.append(f"Velocity: {len(all_txns)} transaction(s) recorded")
        lines.append("")

        lines.append("Centrality measures:")
        lines.append(f"  PageRank: {feats['pagerank']:.5f}")
        lines.append(f"  Degree centrality: {feats['degree_centrality']:.5f}")
        lines.append(f"  Betweenness centrality: {feats['betweenness_centrality']:.5f}")
        lines.append(f"  Clustering coefficient: {feats['clustering_coeff']:.5f}")
        lines.append("")

        if connected_suspicious:
            lines.append(
                f"Connected suspicious accounts ({len(connected_suspicious)}): "
                f"{sorted(connected_suspicious)[:10]}"
            )
        else:
            lines.append("Connected suspicious accounts: none identified via cycle search.")
        lines.append("")

        lines.append("Recent transactions:")
        for _, row in all_txns.tail(8).iterrows():
            direction = "OUT" if row["account_id"] == account_id else "IN"
            other = row["counterparty_id"] if direction == "OUT" else row["account_id"]
            lines.append(
                f"  [{direction}] {row['timestamp']:%Y-%m-%d %H:%M} "
                f"{CURRENCY_SYMBOL}{row['amount']:,.2f} <-> {other} "
                f"({row['transaction_type']}, {row['country']})"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("GRAPH-BASED AML ANALYTICS PLATFORM")
    print("=" * 72)

    # --- 1. Generate data --------------------------------------------------
    print("\n[1/6] Generating synthetic transaction data...")
    generator = TransactionGenerator()
    txns = generator.generate()
    print(f"  Accounts generated              : {len(generator.accounts)}")
    print(f"  Transactions generated          : {len(txns)}")
    print(
        f"  Ground-truth suspicious accounts (from injected patterns): "
        f"{len(generator.suspicious_accounts)}"
    )
    for pattern, members in generator.pattern_membership.items():
        print(f"    - {pattern:<14}: {len(members)} accounts")

    # --- 2. Build graph ------------------------------------------------------
    print("\n[2/6] Building account transaction graph...")
    tx_graph = TransactionGraph(txns)
    simple = tx_graph.simple_graph()
    print(f"  Nodes (accounts)                : {simple.number_of_nodes()}")
    print(f"  Edges (unique account pairs)    : {simple.number_of_edges()}")
    components = tx_graph.connected_components()
    print(f"  Weakly connected components     : {len(components)}")
    largest_cc = max(components, key=len)
    print(f"  Largest connected component size: {len(largest_cc)}")

    # --- 3. Feature engineering -----------------------------------------------
    print("\n[3/6] Engineering graph-level and node-level features...")
    fe = GraphFeatureEngineer(tx_graph)
    features_df = fe.build_features()
    print(f"  Feature table shape             : {features_df.shape}")

    labels = features_df["account_id"].isin(generator.suspicious_accounts).astype(int)
    print(f"  Positive label rate             : {labels.mean():.2%}")

    # --- 4. Scoring & ML -------------------------------------------------------
    print("\n[4/6] Scoring with rule-based system and training ML models...")
    scorer = AMLScorer(features_df)
    results, importances = scorer.train_and_evaluate(labels)

    print("\n  Model comparison (held-out test set):")
    header = f"  {'Model':<22}{'Precision':>10}{'Recall':>10}{'F1':>8}{'PR-AUC':>9}{'P@K':>8}{'Alerts':>8}"
    print(header)
    for name, r in results.items():
        print(
            f"  {name:<22}{r['precision']:>10.3f}{r['recall']:>10.3f}{r['f1']:>8.3f}"
            f"{r['pr_auc']:>9.3f}{r['precision_at_k']:>8.3f}{r['alerts_generated']:>8d}"
        )

    print("\n  Confusion matrices [[TN, FP], [FN, TP]]:")
    for name, r in results.items():
        print(f"    {name:<22}: {r['confusion_matrix']}")

    print("\n  Top 8 most important graph features (Random Forest):")
    for feat, imp in importances.head(8).items():
        print(f"    {feat:<28} {imp:.4f}")

    n_accounts = len(features_df)
    best_model_name = max(results, key=lambda k: results[k]["f1"])
    best = results[best_model_name]
    pct_flagged = best["alerts_generated"] / len(scorer.y_test) * 100

    # --- 5. Investigation ------------------------------------------------------
    print(f"\n[5/6] Investigation workflow (using best model: {best_model_name})...")
    risk_scores = pd.Series(
        scorer.rf_model.predict_proba(features_df[AMLScorer.FEATURE_COLS].fillna(0.0))[:, 1],
        index=features_df["account_id"],
    )

    top_account = risk_scores.sort_values(ascending=False).index[0]
    investigator = InvestigationEngine(tx_graph, features_df, risk_scores)
    report = investigator.investigate(top_account)
    print("\n" + "-" * 72)
    print(report)
    print("-" * 72)

    # --- 6. Experiment summary ----------------------------------------------
    print("\n[6/6] Experiment summary")
    print("-" * 72)
    print(f"  Total accounts analysed           : {n_accounts}")
    print(f"  Ground-truth suspicious accounts  : {labels.sum()} ({labels.mean():.1%})")
    print(f"  Best model by F1                  : {best_model_name}")
    print(f"  Best model precision / recall     : {best['precision']:.3f} / {best['recall']:.3f}")
    print(f"  Best model PR-AUC                 : {best['pr_auc']:.3f}")
    print(
        f"  Alerts generated (test set)       : {best['alerts_generated']} of "
        f"{len(scorer.y_test)} accounts ({pct_flagged:.1f}%)"
    )
    print(f"  True positives / False alerts     : {best['true_positives']} / {best['false_positives']}")
    print(f"  Accounts requiring investigation   : {pct_flagged:.1f}% of test-set book")
    print("-" * 72)
    print("NOTE: All data is synthetic. This is a research / portfolio prototype,")
    print("      NOT a production or regulatory-compliant AML system.")


if __name__ == "__main__":
    main()