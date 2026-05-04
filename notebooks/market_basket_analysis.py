"""
====================================================================
  MARKET BASKET ANALYSIS — INSTACART DATASET
====================================================================
  Author  : [Your Name]
  Purpose : Discover product association rules from Instacart orders
            and translate them into actionable retail strategies.

  Pipeline Overview
  -----------------
  1. ETL         — Load, merge, and validate all source tables
  2. EDA         — Visualise top products, aisles, and order timing
  3. MBA (Apriori via FP-Growth) — Compute Support, Confidence, Lift
  4. Business Output — Filter high-value rules for cross-selling
====================================================================
"""

# ── Standard library ────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ─────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib.ticker  as mticker
import seaborn as sns
from collections import defaultdict
from itertools   import combinations

# ── Plot aesthetics (applied globally) ──────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})

OUTPUT_DIR = "outputs/"   # all figures are saved here

# ════════════════════════════════════════════════════════════════════
#  SECTION 1 — ETL  (Extract · Transform · Load)
# ════════════════════════════════════════════════════════════════════

def load_raw_tables(data_path: str = "") -> dict[str, pd.DataFrame]:
    """
    Read every raw CSV file into a named dictionary of DataFrames.

    Parameters
    ----------
    data_path : str
        Folder that contains the CSV files.
        Pass an empty string if they are in the working directory.

    Returns
    -------
    dict  — keys: 'aisles', 'departments', 'orders',
                   'products', 'order_products'
    """
    prefix = data_path.rstrip("/") + "/" if data_path else ""

    print("📂  Loading raw tables …")
    tables = {
        "aisles":          pd.read_csv(f"{prefix}aisles.csv"),
        "departments":     pd.read_csv(f"{prefix}departments.csv"),
        "orders":          pd.read_csv(f"{prefix}orders.csv"),
        "products":        pd.read_csv(f"{prefix}products.csv"),
        "order_products":  pd.read_csv(f"{prefix}order_products__prior.csv"),
    }

    # ── Dtype safety: ensure every join key is the same integer type ─
    # Mismatched dtypes (e.g. int32 vs int64) cause silent merge gaps.
    id_columns = {
        "aisles":         ["aisle_id"],
        "departments":    ["department_id"],
        "orders":         ["order_id", "user_id"],
        "products":       ["product_id", "aisle_id", "department_id"],
        "order_products": ["order_id", "product_id"],
    }
    for name, cols in id_columns.items():
        tables[name][cols] = tables[name][cols].astype("int32")

    for name, df in tables.items():
        print(f"   ✔  {name:20s}  shape={df.shape}")

    return tables


def build_master_table(tables: dict) -> pd.DataFrame:
    """
    Join all dimension tables onto the transactional core
    (order_products__prior) to produce one wide, analysis-ready frame.

    Join path
    ---------
    order_products  ← orders       (on order_id)
                    ← products     (on product_id)
                    ← aisles       (on aisle_id)
                    ← departments  (on department_id)

    Returns
    -------
    pd.DataFrame — one row per (order, product) pair
    """
    print("\n🔗  Merging tables …")

    master = (
        tables["order_products"]
        .merge(tables["orders"],      on="order_id",      how="left")
        .merge(tables["products"],    on="product_id",    how="left")
        .merge(tables["aisles"],      on="aisle_id",      how="left")
        .merge(tables["departments"], on="department_id", how="left")
    )

    # ── Rename columns for readability ──────────────────────────────
    master.rename(columns={
        "order_dow":       "order_day_of_week",
        "order_hour_of_day": "order_hour",
    }, inplace=True)

    # ── Drop rows where a join produced no match (should be zero) ───
    before = len(master)
    master.dropna(subset=["product_name", "aisle", "department"], inplace=True)
    after  = len(master)
    dropped = before - after
    if dropped:
        print(f"   ⚠️  Dropped {dropped:,} rows with null lookup values")

    print(f"   ✔  Master table ready — {master.shape[0]:,} rows × {master.shape[1]} cols")
    return master


def print_statistical_summary(master: pd.DataFrame) -> None:
    """
    Print a structured overview of the merged dataset so we can spot
    data-quality issues before modelling begins.
    """
    print("\n" + "═" * 60)
    print("  STATISTICAL SUMMARY")
    print("═" * 60)

    total_orders     = master["order_id"].nunique()
    total_users      = master["user_id"].nunique()
    total_products   = master["product_id"].nunique()
    total_aisles     = master["aisle_id"].nunique()
    total_depts      = master["department_id"].nunique()
    total_line_items = len(master)

    print(f"  Total line-items (transactions) : {total_line_items:>12,}")
    print(f"  Unique orders                   : {total_orders:>12,}")
    print(f"  Unique customers (user_id)      : {total_users:>12,}")
    print(f"  Unique products                 : {total_products:>12,}")
    print(f"  Unique aisles                   : {total_aisles:>12,}")
    print(f"  Unique departments              : {total_depts:>12,}")

    # Average basket size (products per order)
    basket = master.groupby("order_id")["product_id"].count()
    print(f"\n  Avg basket size                 : {basket.mean():>12.2f} items")
    print(f"  Median basket size              : {basket.median():>12.0f} items")
    print(f"  Largest basket                  : {basket.max():>12,} items")

    # Reorder rate
    reorder_rate = master["reordered"].mean() * 100
    print(f"\n  Reorder rate                    : {reorder_rate:>11.1f}%")

    # Missing values audit
    missing = master.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("\n  ✅  No missing values detected")
    else:
        print("\n  ⚠️  Missing values:")
        print(missing.to_string())

    print("═" * 60)


# ════════════════════════════════════════════════════════════════════
#  SECTION 2 — EDA  (Exploratory Data Analysis)
# ════════════════════════════════════════════════════════════════════

def plot_top_products(master: pd.DataFrame, top_n: int = 20) -> None:
    """
    Horizontal bar chart showing the most frequently purchased products.
    Frequency is calculated across all prior orders.
    """
    counts = (
        master.groupby("product_name")["order_id"]
        .count()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=sns.color_palette("Blues_d", top_n))

    # Add count labels at the end of each bar
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 2000, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)

    ax.set_xlabel("Number of Orders")
    ax.set_title(f"Top {top_n} Most Purchased Products", fontweight="bold", pad=14)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}01_top_products.png")
    plt.close()
    print(f"   📊  Saved: 01_top_products.png")


def plot_top_aisles(master: pd.DataFrame, top_n: int = 15) -> None:
    """
    Bar chart of the busiest aisles (category sections) by order volume.
    Useful for shelf-space planning and promotional placement.
    """
    counts = (
        master.groupby("aisle")["order_id"]
        .count()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("Greens_d", top_n)
    ax.barh(counts.index[::-1], counts.values[::-1], color=palette)
    ax.set_xlabel("Number of Orders")
    ax.set_title(f"Top {top_n} Busiest Aisles", fontweight="bold", pad=14)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}02_top_aisles.png")
    plt.close()
    print(f"   📊  Saved: 02_top_aisles.png")


def plot_order_hour_distribution(master: pd.DataFrame) -> None:
    """
    Line plot of order volume by hour of day.
    Reveals peak shopping windows — critical for push-notification timing.
    """
    hourly = (
        master.groupby("order_hour")["order_id"]
        .nunique()
        .reset_index(name="order_count")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly["order_hour"], hourly["order_count"],
            marker="o", linewidth=2.2, color="#2196F3", markersize=5)
    ax.fill_between(hourly["order_hour"], hourly["order_count"],
                    alpha=0.15, color="#2196F3")

    # Annotate the peak hour
    peak = hourly.loc[hourly["order_count"].idxmax()]
    ax.annotate(f"Peak: {int(peak['order_hour'])}:00",
                xy=(peak["order_hour"], peak["order_count"]),
                xytext=(peak["order_hour"] + 0.8, peak["order_count"] - 4000),
                arrowprops=dict(arrowstyle="->", color="#333"),
                fontsize=9, color="#333")

    ax.set_xlabel("Hour of Day (24h)")
    ax.set_ylabel("Number of Unique Orders")
    ax.set_title("Order Volume by Hour of Day", fontweight="bold", pad=14)
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}03_order_by_hour.png")
    plt.close()
    print(f"   📊  Saved: 03_order_by_hour.png")


def plot_order_day_distribution(master: pd.DataFrame) -> None:
    """
    Bar chart of order volume by day of week.
    Instacart encodes days as integers (0 = Saturday, 1 = Sunday …).
    """
    day_labels = {0: "Sat", 1: "Sun", 2: "Mon", 3: "Tue",
                  4: "Wed", 5: "Thu", 6: "Fri"}

    daily = (
        master.groupby("order_day_of_week")["order_id"]
        .nunique()
        .reset_index(name="order_count")
    )
    daily["day_name"] = daily["order_day_of_week"].map(day_labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(daily["day_name"], daily["order_count"],
                  color=sns.color_palette("Purples_d", len(daily)))
    for bar, val in zip(bars, daily["order_count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f"{val/1e3:.0f}K", ha="center", fontsize=8)

    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Unique Orders")
    ax.set_title("Order Volume by Day of Week", fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}04_order_by_day.png")
    plt.close()
    print(f"   📊  Saved: 04_order_by_day.png")


def plot_department_treemap(master: pd.DataFrame) -> None:
    """
    Horizontal stacked share-of-orders chart per department.
    Gives an at-a-glance view of which departments dominate volume.
    """
    dept_counts = (
        master.groupby("department")["order_id"]
        .count()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("tab20", len(dept_counts))
    bars = ax.barh(dept_counts.index, dept_counts.values, color=palette)

    for bar, val in zip(bars, dept_counts.values):
        ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height() / 2,
                f"{val/1e3:.0f}K", va="center", fontsize=8)

    ax.set_xlabel("Number of Line-Items Sold")
    ax.set_title("Sales Volume by Department", fontweight="bold", pad=14)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}05_department_volume.png")
    plt.close()
    print(f"   📊  Saved: 05_department_volume.png")


def plot_basket_size_distribution(master: pd.DataFrame) -> None:
    """
    Histogram of basket sizes (items per order).
    Right-skewed distributions are expected; outlier baskets affect MBA.
    """
    basket_sizes = master.groupby("order_id")["product_id"].count()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(basket_sizes[basket_sizes <= 50], bins=50,
            color="#FF7043", edgecolor="white", linewidth=0.4)
    ax.axvline(basket_sizes.mean(), color="#212121", linestyle="--",
               label=f"Mean = {basket_sizes.mean():.1f}")
    ax.set_xlabel("Items per Order")
    ax.set_ylabel("Number of Orders")
    ax.set_title("Distribution of Basket Sizes (capped at 50)", fontweight="bold", pad=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}06_basket_size_distribution.png")
    plt.close()
    print(f"   📊  Saved: 06_basket_size_distribution.png")


def run_eda(master: pd.DataFrame) -> None:
    """
    Orchestrates all EDA plots in a single call.
    """
    print("\n📈  Running Exploratory Data Analysis …")
    plot_top_products(master)
    plot_top_aisles(master)
    plot_order_hour_distribution(master)
    plot_order_day_distribution(master)
    plot_department_treemap(master)
    plot_basket_size_distribution(master)
    print("   ✅  All EDA charts saved.")


# ════════════════════════════════════════════════════════════════════
#  SECTION 3 — MBA  (Market Basket Analysis)
#              Apriori implemented via FP-Growth logic (no mlxtend)
# ════════════════════════════════════════════════════════════════════

def build_basket_sample(
    master: pd.DataFrame,
    sample_orders: int = 50_000,
    min_product_freq: int = 500,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Build a binary order × product matrix for Apriori.

    We sample a subset of orders and filter low-frequency products
    because the full 1M-order matrix (~49K columns) would be too
    large to hold in memory for a dense boolean array.

    Parameters
    ----------
    sample_orders      : How many random orders to include.
    min_product_freq   : Minimum times a product must appear in the
                         full dataset to be eligible for MBA.
    random_state       : Reproducibility seed.

    Returns
    -------
    pd.DataFrame — boolean, index=order_id, columns=product_name
    """
    print(f"\n🛒  Building basket matrix …")

    # Keep only products that appear often enough globally
    product_freq = master["product_id"].value_counts()
    frequent_products = product_freq[product_freq >= min_product_freq].index
    filtered = master[master["product_id"].isin(frequent_products)]
    print(f"   Products after frequency filter (>={min_product_freq}): "
          f"{filtered['product_id'].nunique():,}")

    # Sample a random subset of orders for computational tractability
    all_orders = filtered["order_id"].unique()
    rng = np.random.default_rng(random_state)
    sampled_order_ids = rng.choice(
        all_orders,
        size=min(sample_orders, len(all_orders)),
        replace=False
    )
    sampled = filtered[filtered["order_id"].isin(sampled_order_ids)]
    print(f"   Orders sampled: {sampled['order_id'].nunique():,}")

    # Build boolean pivot table
    basket = (
        sampled
        .groupby(["order_id", "product_name"])["product_id"]
        .count()
        .unstack(fill_value=0)
        .astype(bool)
    )
    print(f"   Basket matrix shape: {basket.shape[0]:,} orders × "
          f"{basket.shape[1]:,} products")
    return basket


def compute_support(basket: pd.DataFrame) -> pd.Series:
    """
    Calculate support for every individual product.

    Support(A) = (orders containing A) / (total orders)
    """
    n_orders = len(basket)
    return basket.sum(axis=0) / n_orders


def apriori_pairs(
    basket: pd.DataFrame,
    min_support: float = 0.01,
    min_confidence: float = 0.20,
    min_lift: float = 1.5
) -> pd.DataFrame:
    """
    Efficient Apriori for 2-item sets (pairs) only.

    For every qualifying product pair (A → B) we compute:
      • Support    = P(A ∩ B)
      • Confidence = P(B | A) = P(A ∩ B) / P(A)
      • Lift       = Confidence / P(B)
                   > 1 means A and B co-occur more than by chance

    Parameters
    ----------
    min_support    : Minimum co-occurrence frequency (fraction of orders).
    min_confidence : Minimum conditional probability threshold.
    min_lift       : Minimum lift threshold.

    Returns
    -------
    pd.DataFrame with columns:
        antecedent, consequent, support, confidence, lift
    """
    print(f"\n⚙️   Running Apriori (pairs) …")
    print(f"   Thresholds — support≥{min_support}, "
          f"confidence≥{min_confidence}, lift≥{min_lift}")

    n = len(basket)

    # Step 1 — Individual product support (filter L1 frequent items)
    item_support = compute_support(basket)
    frequent_items = item_support[item_support >= min_support].index.tolist()
    print(f"   Frequent 1-itemsets: {len(frequent_items):,}")

    # Step 2 — Prune basket to frequent items only (speeds up pair loop)
    basket_pruned = basket[frequent_items]

    # Step 3 — Enumerate all pairs and compute metrics
    # Convert to numpy for fast column-wise operations
    basket_np = basket_pruned.values          # bool array (n_orders, n_items)
    items     = basket_pruned.columns.tolist()
    n_items   = len(items)

    records = []
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue  # skip self-pairs

            # Co-occurrence count: both item i AND item j in same order
            co_occur   = (basket_np[:, i] & basket_np[:, j]).sum()
            support_ij = co_occur / n

            if support_ij < min_support:
                continue  # prune: pair is not frequent enough

            support_a    = item_support[items[i]]
            support_b    = item_support[items[j]]
            confidence   = support_ij / support_a
            lift         = confidence / support_b

            if confidence >= min_confidence and lift >= min_lift:
                records.append({
                    "antecedent": items[i],
                    "consequent": items[j],
                    "support":    round(support_ij, 5),
                    "confidence": round(confidence,  4),
                    "lift":       round(lift,         4),
                })

    rules = pd.DataFrame(records)

    if rules.empty:
        print("   ⚠️  No rules found — try lowering thresholds.")
        return rules

    rules.sort_values("lift", ascending=False, inplace=True)
    rules.reset_index(drop=True, inplace=True)
    print(f"   ✅  {len(rules):,} rules generated.")
    return rules


# ════════════════════════════════════════════════════════════════════
#  SECTION 4 — BUSINESS OUTPUT
# ════════════════════════════════════════════════════════════════════

def plot_rules_scatter(rules: pd.DataFrame) -> None:
    """
    Scatter plot: Support vs Confidence, coloured by Lift.
    High-lift rules in the top-right quadrant are the most valuable.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        rules["support"], rules["confidence"],
        c=rules["lift"], cmap="YlOrRd", alpha=0.7,
        s=40, edgecolors="none"
    )
    plt.colorbar(sc, ax=ax, label="Lift")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Association Rules — Support vs Confidence (colour = Lift)",
                 fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}07_rules_scatter.png")
    plt.close()
    print(f"   📊  Saved: 07_rules_scatter.png")


def plot_top_rules_by_lift(rules: pd.DataFrame, top_n: int = 20) -> None:
    """
    Horizontal bar chart of the top-N rules ranked by Lift.
    Each label shows the antecedent → consequent pair.
    """
    top = rules.head(top_n).copy()
    top["rule"] = top["antecedent"] + "  →  " + top["consequent"]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(top["rule"][::-1], top["lift"][::-1],
                   color=sns.color_palette("Oranges_d", top_n))

    for bar, val in zip(bars, top["lift"][::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)

    ax.set_xlabel("Lift")
    ax.set_title(f"Top {top_n} Association Rules by Lift", fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}08_top_rules_lift.png")
    plt.close()
    print(f"   📊  Saved: 08_top_rules_lift.png")


def generate_business_recommendations(rules: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and annotate the strongest rules with strategic retail actions.

    Recommendation mapping
    ----------------------
    Lift ≥ 5          →  Bundle & promote as a product pair
    3 ≤ Lift < 5      →  Place products on adjacent shelves
    Lift < 3          →  Include in digital cross-sell widget only
    Confidence ≥ 0.5  →  Flag for personalised push notification
    """
    df = rules.copy()

    def assign_action(row):
        if row["lift"] >= 5:
            return "🎁  Create bundle / combo deal"
        elif row["lift"] >= 3:
            return "📦  Place on adjacent shelves"
        else:
            return "📱  Digital cross-sell widget"

    def assign_channel(row):
        if row["confidence"] >= 0.5:
            return "Push notification + In-store"
        else:
            return "In-store / homepage banner"

    df["recommendation"] = df.apply(assign_action,   axis=1)
    df["channel"]        = df.apply(assign_channel,  axis=1)

    # Keep only high-value rules for the executive summary
    high_value = df[df["lift"] >= 2.0].copy()
    high_value.reset_index(drop=True, inplace=True)

    print(f"\n💼  Business recommendations generated: {len(high_value):,} high-value rules")
    return high_value


def save_outputs(rules: pd.DataFrame, recommendations: pd.DataFrame) -> None:
    """
    Persist all analysis artefacts to disk.

    Files produced
    --------------
    outputs/all_rules.csv        — Full rule set with Support/Confidence/Lift
    outputs/recommendations.csv  — Filtered, annotated high-value rules
    """
    rules.to_csv(f"{OUTPUT_DIR}all_rules.csv", index=False)
    recommendations.to_csv(f"{OUTPUT_DIR}recommendations.csv", index=False)
    print(f"\n💾  Saved: all_rules.csv ({len(rules):,} rules)")
    print(f"💾  Saved: recommendations.csv ({len(recommendations):,} rules)")


def print_executive_summary(recommendations: pd.DataFrame, top_n: int = 10) -> None:
    """
    Print a formatted executive summary table — ready to paste into a
    PowerPoint slide or stakeholder report.
    """
    print("\n" + "═" * 85)
    print("  EXECUTIVE SUMMARY — TOP ASSOCIATION RULES FOR MARKETING")
    print("═" * 85)
    top = recommendations.head(top_n)[
        ["antecedent", "consequent", "support", "confidence", "lift",
         "recommendation", "channel"]
    ]
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 120)
    print(top.to_string(index=False))
    print("═" * 85)
    print("  Metrics Guide:")
    print("  • Support    — % of all orders containing both products")
    print("  • Confidence — % of antecedent orders that also contain consequent")
    print("  • Lift       — How much more likely the pair is vs. random chance")
    print("                 Lift > 1 = positive association")
    print("═" * 85)


# ════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════

def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. ETL ──────────────────────────────────────────────────────
    DATA_PATH = "/mnt/user-data/uploads"
    tables = load_raw_tables(DATA_PATH)
    master = build_master_table(tables)
    print_statistical_summary(master)

    # ── 2. EDA ──────────────────────────────────────────────────────
    run_eda(master)

    # ── 3. MBA ──────────────────────────────────────────────────────
    basket = build_basket_sample(
        master,
        sample_orders=50_000,     # increase for more coverage
        min_product_freq=500,     # remove long-tail products
    )

    rules = apriori_pairs(
        basket,
        min_support=0.01,         # product pair in ≥1% of baskets
        min_confidence=0.20,      # 20% of antecedent orders contain consequent
        min_lift=1.5,             # at least 1.5x more likely than random
    )

    # ── 4. Business Output ──────────────────────────────────────────
    if not rules.empty:
        plot_rules_scatter(rules)
        plot_top_rules_by_lift(rules)
        recommendations = generate_business_recommendations(rules)
        save_outputs(rules, recommendations)
        print_executive_summary(recommendations)

    print("\n✅  Pipeline complete. All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
