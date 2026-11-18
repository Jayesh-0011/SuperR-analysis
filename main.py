import os
import pandas as pd
import numpy as np


# ==========================================
# STEP 1: LOAD DATASETS
# ==========================================

DATA_PATH = "data"
RESULTS_PATH = "results"

os.makedirs(RESULTS_PATH, exist_ok=True)

files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

datasets = {}

for file in files:
    name = file.replace(".csv", "")
    datasets[name] = pd.read_csv(os.path.join(DATA_PATH, file))

    print(f"\nLoaded: {name}")
    print("Shape:", datasets[name].shape)

print("\nAll datasets loaded successfully!")


# ==========================================
# STEP 2: EXPLORE DATASETS
# ==========================================

for name, data in datasets.items():

    print("\n" + "=" * 50)
    print(f"DATASET: {name}")
    print("=" * 50)

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nColumns:")
    print(data.columns.tolist())

    print("\nMissing Values:")
    print(data.isnull().sum())

    print("\nData Types:")
    print(data.dtypes)

    print("\nDuplicate Rows:")
    print(data.duplicated().sum())


# ==========================================
# STEP 3: EXTRACT DATASETS
# ==========================================

ball_by_ball = datasets["ball_by_ball"]
matches = datasets["matches"]


# ==========================================
# STEP 4: MERGE DATASETS
# ==========================================

print("\n========== MERGING DATASETS ==========")

duplicate_match_ids = matches["match_id"].duplicated().sum()

print("Duplicate match IDs:", duplicate_match_ids)

if duplicate_match_ids > 0:
    raise ValueError("Duplicate match IDs found. Check the matches dataset.")

merged_df = pd.merge(
    ball_by_ball,
    matches,
    on="match_id",
    how="left",
    validate="many_to_one",
    indicator=True
)

print("\nOriginal dataset shape:", ball_by_ball.shape)
print("Merged dataset shape:", merged_df.shape)

print("\nMerge status:")
print(merged_df["_merge"].value_counts())

unmatched = merged_df[merged_df["_merge"] == "left_only"]

print("\nUnmatched records:", len(unmatched))

if len(unmatched) > 0:
    raise ValueError("Some deliveries could not be matched with match information.")

merged_df = merged_df.drop(columns=["_merge"])

merged_output_file = os.path.join(RESULTS_PATH, "merged_data.csv")
merged_df.to_csv(merged_output_file, index=False)

print("\nMerged dataset saved successfully!")


# ==========================================
# STEP 5: VALIDATE RUNS AND EXTRAS
# ==========================================

print("\n========== EXTRAS VALIDATION ==========")

print("\nExtras type distribution:")
print(ball_by_ball["extras_type"].value_counts(dropna=False))

extras_df = ball_by_ball[
    ball_by_ball["extras1_runs"].ne(0)
    | ball_by_ball["extras_2_runs"].ne(0)
]

print("\nSample deliveries containing extras:")

print(
    extras_df[[
        "match_id",
        "innings",
        "runs",
        "extras_type",
        "extras1_runs",
        "extras_2_runs",
        "is_wicket"
    ]].head(20).to_string(index=False)
)

print("\nRuns and extras summary:")

print(
    ball_by_ball[[
        "runs",
        "extras1_runs",
        "extras_2_runs"
    ]].describe()
)


# ==========================================
# STEP 6: RECONSTRUCT MATCH SITUATION
# ==========================================

# IMPORTANT:
# All match-situation calculations are performed on df.
# Therefore, later steps must also use df, not merged_df.

df = merged_df.copy()


# ------------------------------------------
# 6.1 Calculate total runs
# ------------------------------------------

# runs = batter runs
# extras1_runs and extras_2_runs = additional runs

df["total_runs"] = (
    df["runs"]
    + df["extras1_runs"].fillna(0)
    + df["extras_2_runs"].fillna(0)
)


# ------------------------------------------
# 6.2 Identify illegal deliveries
# ------------------------------------------

extras = df["extras_type"].fillna("").str.lower().str.strip()

# Keeping the current classification:
# Wides and Overstep (no-ball) are illegal deliveries.
df["is_illegal_delivery"] = extras.isin([
    "wide",
    "overstep"
])

df["legal_ball"] = (
    ~df["is_illegal_delivery"]
).astype(int)


# ------------------------------------------
# 6.3 Sort deliveries chronologically
# ------------------------------------------

df = df.sort_values(
    by=[
        "match_id",
        "innings",
        "innings_ball_num"
    ]
).reset_index(drop=True)


# ------------------------------------------
# 6.4 Calculate cumulative match situation
# ------------------------------------------

group_columns = ["match_id", "innings"]

# Cumulative score
df["current_score"] = (
    df.groupby(group_columns)["total_runs"].cumsum()
)

# Cumulative wickets
df["wickets_lost"] = (
    df.groupby(group_columns)["is_wicket"].cumsum()
)

# Cumulative legal deliveries
df["legal_balls_bowled"] = (
    df.groupby(group_columns)["legal_ball"].cumsum()
)

# Overs completed
df["overs_completed"] = (
    df["legal_balls_bowled"] // 6
)

# Balls in current over
df["balls_in_current_over"] = (
    df["legal_balls_bowled"] % 6
)

# Current run rate
df["current_run_rate"] = (
    df["current_score"] * 6
    / df["legal_balls_bowled"].replace(0, np.nan)
)


# ------------------------------------------
# 6.5 Calculate first innings total
# ------------------------------------------

first_innings_totals = (
    df[df["innings"] == 1]
    .groupby("match_id")["total_runs"]
    .sum()
    .rename("first_innings_total")
)

df = df.merge(
    first_innings_totals,
    on="match_id",
    how="left",
    validate="many_to_one"
)


# ------------------------------------------
# 6.6 Calculate target
# ------------------------------------------

# Use revised target when available.
# Otherwise, target = first innings total + 1.

df["target"] = np.where(
    df["revised_target"].notna(),
    df["revised_target"],
    df["first_innings_total"] + 1
)


# ------------------------------------------
# 6.7 Calculate runs required
# ------------------------------------------

df["runs_required"] = (
    df["target"] - df["current_score"]
).clip(lower=0)


# ------------------------------------------
# 6.8 Calculate balls remaining
# ------------------------------------------

# Default T20 innings length = 20 overs = 120 legal balls.

df["innings_overs"] = 20.0

# Use revised overs when available.
# Second innings revised overs apply to innings 2.
# First innings revised overs apply to innings 1.

df.loc[
    (df["innings"] == 1)
    & df["first_inn_revised_overs"].notna(),
    "innings_overs"
] = df["first_inn_revised_overs"]

df.loc[
    (df["innings"] == 2)
    & df["second_inn_revised_overs"].notna(),
    "innings_overs"
] = df["second_inn_revised_overs"]

df["total_legal_balls"] = (
    df["innings_overs"] * 6
).round().astype(int)

df["balls_remaining"] = (
    df["total_legal_balls"] - df["legal_balls_bowled"]
).clip(lower=0)


# ------------------------------------------
# 6.9 Calculate required run rate
# ------------------------------------------

df["required_run_rate"] = np.where(
    (df["innings"] == 2)
    & (df["balls_remaining"] > 0),
    df["runs_required"] * 6 / df["balls_remaining"],
    np.nan
)


# ==========================================
# STEP 6.10: VALIDATE MATCH SITUATION
# ==========================================

print("\n========== SECOND INNINGS SAMPLE ==========")

second_innings = df[df["innings"] == 2]

print(
    second_innings[[
        "match_id",
        "innings_ball_num",
        "total_runs",
        "current_score",
        "wickets_lost",
        "legal_balls_bowled",
        "target",
        "runs_required",
        "balls_remaining",
        "required_run_rate"
    ]].head(20).to_string(index=False)
)


# ==========================================
# CHECK FINAL INNINGS SCORES
# ==========================================

print("\n========== FINAL INNINGS SCORES ==========")

innings_summary = (
    df.groupby(["match_id", "innings"])
    .agg(
        final_score=("total_runs", "sum"),
        wickets_lost=("is_wicket", "sum"),
        legal_balls=("legal_ball", "sum")
    )
    .reset_index()
)

print(innings_summary.head(20).to_string(index=False))


# ==========================================
# CHECK UNUSUAL EXTRAS
# ==========================================

print("\n========== OTHER AND PENALTY EXTRAS ==========")

unusual_extras = ball_by_ball[
    ball_by_ball["extras_type"].isin(["other", "Penalty"])
]

print(
    unusual_extras[[
        "match_id",
        "innings",
        "overid",
        "over_ball_num",
        "innings_ball_num",
        "runs",
        "extras_type",
        "extras1_runs",
        "extras_2_runs",
        "is_wicket",
        "dismissal_type"
    ]].to_string(index=False)
)


# ==========================================
# STEP 7: IDENTIFY PRESSURE SITUATIONS
# ==========================================

# Pressure is assessed only during second-innings chases.
# Wicket loss is NOT used as a pressure criterion.
#
# Condition 1: Required run rate >= 10
# Condition 2: 30 or fewer legal balls remain
#
# 0 conditions = Low pressure
# 1 condition  = Moderate pressure
# 2 conditions = High pressure

# Use df here because required_run_rate and balls_remaining
# were calculated in df in Step 6.

df["is_chase"] = df["innings"] == 2

# Condition 1: Required run rate is high
df["rrr_pressure"] = (
    df["is_chase"]
    & (df["required_run_rate"] >= 10)
)

# Condition 2: Thirty or fewer balls remain
df["time_pressure"] = (
    df["is_chase"]
    & (df["balls_remaining"] <= 30)
)

# Count satisfied pressure conditions
df["pressure_factors"] = (
    df["rrr_pressure"].astype(int)
    + df["time_pressure"].astype(int)
)

# Classify pressure situations
df["pressure_level"] = "Not Applicable"

df.loc[
    df["is_chase"] & (df["pressure_factors"] == 0),
    "pressure_level"
] = "Low"

df.loc[
    df["is_chase"] & (df["pressure_factors"] == 1),
    "pressure_level"
] = "Moderate"

df.loc[
    df["is_chase"] & (df["pressure_factors"] == 2),
    "pressure_level"
] = "High"


# ==========================================
# STEP 8: VALIDATE PRESSURE CLASSIFICATION
# ==========================================

print("\n========== PRESSURE LEVEL DISTRIBUTION ==========")
print(df["pressure_level"].value_counts())

print("\n========== PRESSURE LEVEL BY INNINGS ==========")
print(
    pd.crosstab(
        df["innings"],
        df["pressure_level"]
    )
)

print("\n========== SAMPLE HIGH-PRESSURE DELIVERIES ==========")

high_pressure_deliveries = df[
    df["pressure_level"] == "High"
]

print(
    high_pressure_deliveries[[
        "match_id",
        "innings",
        "striker_name",
        "bowler_name",
        "current_score",
        "runs_required",
        "balls_remaining",
        "required_run_rate",
        "pressure_factors",
        "pressure_level"
    ]].head(20).to_string(index=False)
)


# ==========================================
# STEP 9: SAVE FINAL DATASETS
# ==========================================

match_situation_file = os.path.join(
    RESULTS_PATH,
    "match_situation.csv"
)

pressure_analysis_file = os.path.join(
    RESULTS_PATH,
    "pressure_analysis.csv"
)

# Save match situation calculations
df.to_csv(match_situation_file, index=False)

# Save the same dataset with pressure classifications
df.to_csv(pressure_analysis_file, index=False)

print("\n========== PROCESS COMPLETED ==========")
print("Final dataset shape:", df.shape)
print("Match situation file saved at:", match_situation_file)
print("Pressure analysis file saved at:", pressure_analysis_file)
print("\nAll steps completed successfully!")



# ==========================================
# STEP 10: BATTER PERFORMANCE ANALYSIS
# ==========================================

# IMPORTANT METHODOLOGY NOTE:
# Pressure must describe the situation BEFORE a delivery,
# because the batter's response to that delivery is what we evaluate.
#
# The earlier pressure_level column described the situation AFTER
# each delivery. We therefore calculate a pre-delivery pressure
# classification here and use it for batting analysis.


# ------------------------------------------
# 10.1 Keep ordinary T20 innings
# ------------------------------------------

batting_df = df[df["innings"].isin([1, 2])].copy()


# ------------------------------------------
# 10.2 Reconstruct the situation BEFORE each delivery
# ------------------------------------------

batting_group = ["match_id", "innings"]

# Score and legal balls completed before the current delivery
batting_df["score_before_ball"] = (
    batting_df["current_score"] - batting_df["total_runs"]
)

batting_df["legal_balls_before_ball"] = (
    batting_df["legal_balls_bowled"] - batting_df["legal_ball"]
)

batting_df["balls_remaining_before_ball"] = (
    batting_df["total_legal_balls"]
    - batting_df["legal_balls_before_ball"]
).clip(lower=0)

batting_df["runs_required_before_ball"] = (
    batting_df["target"] - batting_df["score_before_ball"]
).clip(lower=0)

batting_df["required_run_rate_before_ball"] = np.where(
    (batting_df["innings"] == 2)
    & (batting_df["balls_remaining_before_ball"] > 0),
    (
        batting_df["runs_required_before_ball"] * 6
        / batting_df["balls_remaining_before_ball"]
    ),
    np.nan
)


# ------------------------------------------
# 10.3 Classify pressure BEFORE the delivery
# ------------------------------------------

batting_df["is_chase"] = batting_df["innings"] == 2

batting_df["rrr_pressure"] = (
    batting_df["is_chase"]
    & (batting_df["required_run_rate_before_ball"] >= 10)
)

batting_df["time_pressure"] = (
    batting_df["is_chase"]
    & (batting_df["balls_remaining_before_ball"] <= 30)
)

batting_df["pressure_factors"] = (
    batting_df["rrr_pressure"].astype(int)
    + batting_df["time_pressure"].astype(int)
)

batting_df["pressure_level"] = "Not Applicable"

batting_df.loc[
    batting_df["is_chase"] & (batting_df["pressure_factors"] == 0),
    "pressure_level"
] = "Low"

batting_df.loc[
    batting_df["is_chase"] & (batting_df["pressure_factors"] == 1),
    "pressure_level"
] = "Moderate"

batting_df.loc[
    batting_df["is_chase"] & (batting_df["pressure_factors"] == 2),
    "pressure_level"
] = "High"


# ------------------------------------------
# 10.4 Define batting statistics
# ------------------------------------------

# Batter runs are stored in the 'runs' column.
batting_df["batter_runs"] = batting_df["runs"].fillna(0)

# A wide is not counted as a ball faced.
# A no-ball is counted as a ball faced in batting statistics.
extras_clean = (
    batting_df["extras_type"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

batting_df["balls_faced"] = (extras_clean != "wide").astype(int)

# Count dismissals against the batter who was actually dismissed.
# This avoids charging a run-out of the non-striker to the striker.
batting_df["batter_dismissed"] = (
    batting_df["is_wicket"].fillna(False)
    & batting_df["dismissed_player_id"].eq(batting_df["striker_id"])
)

batting_df["fours"] = batting_df["is_four"].fillna(False).astype(int)
batting_df["sixes"] = batting_df["is_six"].fillna(False).astype(int)

batting_df["dot_balls"] = (
    (batting_df["batter_runs"] == 0)
    & (batting_df["balls_faced"] == 1)
).astype(int)


# ------------------------------------------
# 10.5 Create a reusable aggregation function
# ------------------------------------------

def calculate_batting_stats(data, group_columns):
    """Calculate batting statistics for the supplied groups."""

    summary = (
        data.groupby(group_columns)
        .agg(
            runs=("batter_runs", "sum"),
            balls_faced=("balls_faced", "sum"),
            dismissals=("batter_dismissed", "sum"),
            fours=("fours", "sum"),
            sixes=("sixes", "sum"),
            dot_balls=("dot_balls", "sum")
        )
        .reset_index()
    )

    summary["strike_rate"] = np.where(
        summary["balls_faced"] > 0,
        summary["runs"] * 100 / summary["balls_faced"],
        np.nan
    )

    summary["batting_average"] = np.where(
        summary["dismissals"] > 0,
        summary["runs"] / summary["dismissals"],
        np.nan
    )

    summary["boundary_runs"] = (
        summary["fours"] * 4 + summary["sixes"] * 6
    )

    summary["boundary_percentage"] = np.where(
        summary["runs"] > 0,
        summary["boundary_runs"] * 100 / summary["runs"],
        0
    )

    summary["dot_ball_percentage"] = np.where(
        summary["balls_faced"] > 0,
        summary["dot_balls"] * 100 / summary["balls_faced"],
        np.nan
    )

    return summary


# ------------------------------------------
# 10.6 Overall batting performance
# ------------------------------------------

overall_batting = calculate_batting_stats(
    batting_df,
    ["striker_id", "striker_name"]
)

overall_batting = overall_batting.rename(columns={
    "runs": "overall_runs",
    "balls_faced": "overall_balls_faced",
    "dismissals": "overall_dismissals",
    "fours": "overall_fours",
    "sixes": "overall_sixes",
    "dot_balls": "overall_dot_balls",
    "strike_rate": "overall_strike_rate",
    "batting_average": "overall_batting_average",
    "boundary_runs": "overall_boundary_runs",
    "boundary_percentage": "overall_boundary_percentage",
    "dot_ball_percentage": "overall_dot_ball_percentage"
})


# ------------------------------------------
# 10.7 Batting performance by pressure level
# ------------------------------------------

pressure_batting_data = batting_df[
    batting_df["pressure_level"].isin(["Low", "Moderate", "High"])
].copy()

pressure_batting = calculate_batting_stats(
    pressure_batting_data,
    ["striker_id", "striker_name", "pressure_level"]
)

# Save detailed pressure-wise statistics
pressure_batting_file = os.path.join(
    RESULTS_PATH,
    "batting_by_pressure.csv"
)

pressure_batting.to_csv(pressure_batting_file, index=False)


# ------------------------------------------
# 10.8 Compare high-pressure performance with
#      non-high-pressure chase performance
# ------------------------------------------

# Low + Moderate pressure deliveries form the comparison baseline.
non_high_pressure_data = pressure_batting_data[
    pressure_batting_data["pressure_level"].isin(["Low", "Moderate"])
]

non_high_batting = calculate_batting_stats(
    non_high_pressure_data,
    ["striker_id", "striker_name"]
)

non_high_batting = non_high_batting.rename(columns={
    "runs": "non_high_pressure_runs",
    "balls_faced": "non_high_pressure_balls",
    "dismissals": "non_high_pressure_dismissals",
    "fours": "non_high_pressure_fours",
    "sixes": "non_high_pressure_sixes",
    "dot_balls": "non_high_pressure_dot_balls",
    "strike_rate": "non_high_pressure_strike_rate",
    "batting_average": "non_high_pressure_average",
    "boundary_runs": "non_high_pressure_boundary_runs",
    "boundary_percentage": "non_high_pressure_boundary_percentage",
    "dot_ball_percentage": "non_high_pressure_dot_ball_percentage"
})

high_pressure_batting = pressure_batting[
    pressure_batting["pressure_level"] == "High"
].copy()

high_pressure_batting = high_pressure_batting.rename(columns={
    "runs": "high_pressure_runs",
    "balls_faced": "high_pressure_balls",
    "dismissals": "high_pressure_dismissals",
    "fours": "high_pressure_fours",
    "sixes": "high_pressure_sixes",
    "dot_balls": "high_pressure_dot_balls",
    "strike_rate": "high_pressure_strike_rate",
    "batting_average": "high_pressure_average",
    "boundary_runs": "high_pressure_boundary_runs",
    "boundary_percentage": "high_pressure_boundary_percentage",
    "dot_ball_percentage": "high_pressure_dot_ball_percentage"
})

high_pressure_batting = high_pressure_batting.drop(
    columns=["pressure_level"]
)

batting_comparison = high_pressure_batting.merge(
    non_high_batting,
    on=["striker_id", "striker_name"],
    how="left",
    validate="one_to_one"
)

batting_comparison = batting_comparison.merge(
    overall_batting,
    on=["striker_id", "striker_name"],
    how="left",
    validate="one_to_one"
)

# Compare high-pressure strike rate against the batter's overall
# strike rate across the dataset's first and second innings.
# This is the player's dataset-wide baseline, not necessarily
# their full career across every competition.
batting_comparison["strike_rate_difference"] = (
    batting_comparison["high_pressure_strike_rate"]
    - batting_comparison["overall_strike_rate"]
)

batting_comparison["average_difference"] = (
    batting_comparison["high_pressure_average"]
    - batting_comparison["non_high_pressure_average"]
)


# ------------------------------------------
# 10.9 Apply a minimum high-pressure sample
# ------------------------------------------

# A player must have faced at least 20 high-pressure balls
# to be included in the initial comparison shortlist.
MIN_HIGH_PRESSURE_BALLS = 20

qualified_batters = batting_comparison[
    batting_comparison["high_pressure_balls"]
    >= MIN_HIGH_PRESSURE_BALLS
].copy()

# Sort by high-pressure strike rate for an initial inspection.
# This is NOT yet the final reliability ranking.
qualified_batters = qualified_batters.sort_values(
    by=[
        "strike_rate_difference",
        "high_pressure_strike_rate",
        "high_pressure_balls"
    ],
    ascending=[False, False, False]
)


# ------------------------------------------
# 10.10 Display results
# ------------------------------------------

print("\n========== OVERALL BATTING PERFORMANCE ==========")

print(
    overall_batting.sort_values(
        "overall_runs",
        ascending=False
    ).head(20).to_string(index=False)
)

print("\n========== BATTING PERFORMANCE BY PRESSURE ==========")

print(
    pressure_batting.sort_values(
        ["pressure_level", "runs"],
        ascending=[True, False]
    ).head(30).to_string(index=False)
)

print("\n========== HIGH-PRESSURE BATTERS: MINIMUM SAMPLE APPLIED ==========")

print("Minimum high-pressure balls:", MIN_HIGH_PRESSURE_BALLS)
print("Qualified batters:", len(qualified_batters))

display_columns = [
    "striker_id",
    "striker_name",
    "high_pressure_runs",
    "high_pressure_balls",
    "high_pressure_strike_rate",
    "non_high_pressure_runs",
    "non_high_pressure_balls",
    "non_high_pressure_strike_rate",
    "strike_rate_difference",
    "high_pressure_average",
    "non_high_pressure_average",
    "average_difference"
]

if not qualified_batters.empty:
    print(
        qualified_batters[display_columns]
        .head(20)
        .to_string(index=False)
    )
else:
    print(
        "No batters meet the minimum high-pressure balls threshold. "
        "Consider reviewing the threshold."
    )


# ------------------------------------------
# 10.11 Save batting analysis files
# ------------------------------------------

overall_batting_file = os.path.join(
    RESULTS_PATH,
    "overall_batting.csv"
)

batting_comparison_file = os.path.join(
    RESULTS_PATH,
    "batting_pressure_comparison.csv"
)

qualified_batters_file = os.path.join(
    RESULTS_PATH,
    "qualified_high_pressure_batters.csv"
)

# Save the delivery-level data with pre-delivery pressure labels.
batting_delivery_file = os.path.join(
    RESULTS_PATH,
    "batting_deliveries_with_pressure.csv"
)

overall_batting.to_csv(overall_batting_file, index=False)
batting_comparison.to_csv(batting_comparison_file, index=False)
qualified_batters.to_csv(qualified_batters_file, index=False)
batting_df.to_csv(batting_delivery_file, index=False)

print("\n========== BATTING ANALYSIS COMPLETED ==========")
print("Overall batting:", overall_batting_file)
print("Pressure-wise batting:", pressure_batting_file)
print("Pressure comparison:", batting_comparison_file)
print("Qualified high-pressure batters:", qualified_batters_file)
print("Delivery-level pressure data:", batting_delivery_file)


# ==========================================
# STEP 11: RELIABILITY + WINNING/NOT-OUT BATTER RANKING
# ==========================================

# Ranking components:
# 70% = reliability-adjusted high-pressure batting performance
# 20% = percentage of high-pressure innings where the batter's
#        team won AND the batter remained not out
# 10% = percentage of high-pressure innings where the batter's
#        team won AND the batter was dismissed
# All three components are converted to percentile ranks (0-100).

print("\n========== HIGH-PRESSURE BATTER RANKING ==========")

MIN_HIGH_PRESSURE_BALLS = 50
MIN_NON_HIGH_PRESSURE_BALLS = 50
MIN_HIGH_PRESSURE_MATCHES = 5
RELIABILITY_PRIOR_BALLS = 50

PERFORMANCE_WEIGHT = 0.70
WIN_NOT_OUT_WEIGHT = 0.20
WIN_OUT_WEIGHT = 0.10

# ------------------------------------------
# 11.1 Filter batters by minimum ball samples
# ------------------------------------------

reliable_batters = batting_comparison[
    (batting_comparison["high_pressure_balls"] >= MIN_HIGH_PRESSURE_BALLS)
    & (batting_comparison["non_high_pressure_balls"] >= MIN_NON_HIGH_PRESSURE_BALLS)
].copy()

# ------------------------------------------
# 11.2 Calculate reliability-adjusted batting
#      improvement over overall strike rate
# ------------------------------------------

high_sample_factor = np.sqrt(
    reliable_batters["high_pressure_balls"]
    / (reliable_batters["high_pressure_balls"] + RELIABILITY_PRIOR_BALLS)
)

non_high_sample_factor = np.sqrt(
    reliable_batters["non_high_pressure_balls"]
    / (reliable_batters["non_high_pressure_balls"] + RELIABILITY_PRIOR_BALLS)
)

reliable_batters["reliability_score"] = (
    reliable_batters["strike_rate_difference"]
    * high_sample_factor
    * non_high_sample_factor
)

# ------------------------------------------
# 11.3 Identify each batter's pressure innings
#      and classify the combined match outcome
# ------------------------------------------

# Each batter-match-innings combination is counted once, even if
# the batter faced several high-pressure deliveries in that innings.
pressure_innings = (
    batting_df[batting_df["pressure_level"] == "High"]
    [["striker_id", "striker_name", "match_id", "innings",
      "batting_team_id", "win_team_id"]]
    .drop_duplicates()
    .copy()
)

# Find whether the batter was dismissed at any point in that innings.
# This includes run-outs and other dismissals, not only striker dismissals.
innings_dismissals = (
    batting_df[batting_df["is_wicket"] & batting_df["dismissed_player_id"].notna()]
    .groupby(["match_id", "innings", "dismissed_player_id"])
    .size()
    .reset_index(name="dismissal_count")
)

pressure_innings = pressure_innings.merge(
    innings_dismissals,
    left_on=["match_id", "innings", "striker_id"],
    right_on=["match_id", "innings", "dismissed_player_id"],
    how="left"
)

pressure_innings["remained_not_out"] = (
    pressure_innings["dismissal_count"].isna()
).astype(int)

pressure_innings["team_won_match"] = (
    pressure_innings["batting_team_id"].notna()
    & pressure_innings["win_team_id"].notna()
    & (pressure_innings["batting_team_id"] == pressure_innings["win_team_id"])
)

# Mutually exclusive outcomes: winning + not out, or winning + out.
pressure_innings["winning_and_not_out"] = (
    pressure_innings["team_won_match"]
    & (pressure_innings["remained_not_out"] == 1)
).astype(int)

pressure_innings["winning_and_out"] = (
    pressure_innings["team_won_match"]
    & (pressure_innings["remained_not_out"] == 0)
).astype(int)

# Aggregate pressure-innings outcomes for each batter.
pressure_outcome_stats = (
    pressure_innings
    .groupby(["striker_id", "striker_name"])
    .agg(
        high_pressure_innings=("innings", "count"),
        winning_not_out_innings=("winning_and_not_out", "sum"),
        winning_out_innings=("winning_and_out", "sum"),
        high_pressure_matches=("match_id", "nunique")
    )
    .reset_index()
)

# Rates use all pressure innings as the denominator. Matches without a
# recorded winner do not count as either winning category.
pressure_outcome_stats["winning_and_not_out_rate"] = (
    pressure_outcome_stats["winning_not_out_innings"] * 100
    / pressure_outcome_stats["high_pressure_innings"]
)

pressure_outcome_stats["winning_and_out_rate"] = (
    pressure_outcome_stats["winning_out_innings"] * 100
    / pressure_outcome_stats["high_pressure_innings"]
)

reliable_batters = reliable_batters.merge(
    pressure_outcome_stats,
    on=["striker_id", "striker_name"],
    how="inner",
    validate="one_to_one"
)

reliable_batters = reliable_batters[
    reliable_batters["high_pressure_matches"] >= MIN_HIGH_PRESSURE_MATCHES
].copy()

# ------------------------------------------
# 11.4 Convert components to percentile ranks
#      and calculate the final weighted score
# ------------------------------------------

reliable_batters["performance_percentile"] = (
    reliable_batters["reliability_score"].rank(pct=True) * 100
)

reliable_batters["winning_not_out_percentile"] = (
    reliable_batters["winning_and_not_out_rate"].rank(pct=True) * 100
)

reliable_batters["winning_out_percentile"] = (
    reliable_batters["winning_and_out_rate"].rank(pct=True) * 100
)

reliable_batters["final_score"] = (
    PERFORMANCE_WEIGHT * reliable_batters["performance_percentile"]
    + WIN_NOT_OUT_WEIGHT * reliable_batters["winning_not_out_percentile"]
    + WIN_OUT_WEIGHT * reliable_batters["winning_out_percentile"]
)

reliable_batters = reliable_batters.sort_values(
    by=["final_score", "reliability_score", "winning_and_not_out_rate"],
    ascending=[False, False, False]
).reset_index(drop=True)

# ------------------------------------------
# 11.5 Save and display the final ranking
# ------------------------------------------

reliable_batters_file = os.path.join(
    RESULTS_PATH,
    "reliable_high_pressure_batters.csv"
)

reliable_batters.to_csv(reliable_batters_file, index=False)

ranking_columns = [
    "striker_id",
    "striker_name",
    "high_pressure_balls",
    "high_pressure_strike_rate",
    "overall_strike_rate",
    "strike_rate_difference",
    "reliability_score",
    "high_pressure_matches",
    "high_pressure_innings",
    "winning_not_out_innings",
    "winning_and_not_out_rate",
    "winning_out_innings",
    "winning_and_out_rate",
    "performance_percentile",
    "winning_not_out_percentile",
    "winning_out_percentile",
    "final_score"
]

print("\nTop 10 Reliable High-Pressure Batters:\n")

if not reliable_batters.empty:
    print(
        reliable_batters[ranking_columns]
        .head(10)
        .round(2)
        .to_string(index=False)
    )
else:
    print(
        "No batters meet all sample requirements. "
        "Review the ball and match thresholds."
    )

print("\nMinimum high-pressure balls:", MIN_HIGH_PRESSURE_BALLS)
print("Minimum non-high-pressure balls:", MIN_NON_HIGH_PRESSURE_BALLS)
print("Minimum high-pressure matches:", MIN_HIGH_PRESSURE_MATCHES)
print("Performance weight:", PERFORMANCE_WEIGHT * 100, "%")
print("Winning + not-out weight:", WIN_NOT_OUT_WEIGHT * 100, "%")
print("Winning + out weight:", WIN_OUT_WEIGHT * 100, "%")
print("Number of qualified batters:", len(reliable_batters))
print("Saved:", reliable_batters_file)


# ==========================================
# STEP 12: HIGH-PRESSURE BOWLING RANKING
# ==========================================

# Ranking components:
# 90% = Pressure-control performance (percentile rank).
# 10% = Bowler-credited high-pressure wicket-taking rate (percentile rank).
#
# For each bowler in each over:
# Pressure Control = RRR at the start of the over
#                    - economy rate from that bowler's
#                      high-pressure deliveries in that over.
#
# Wicket-taking rate = bowler-credited wickets per 6 legal balls.
# Only wickets credited to the bowler are counted; run-outs and other
# non-bowler dismissals are excluded.

print("\n========== HIGH-PRESSURE BOWLER RANKING ==========")

MIN_HIGH_PRESSURE_BALLS_BOWLED = 35
MIN_HIGH_PRESSURE_MATCHES_BOWLED = 0
# Require a positive average of legal high-pressure balls per match.
MIN_AVG_HIGH_PRESSURE_BALLS_PER_MATCH = 0

# ------------------------------------------
# 12.1 Select high-pressure deliveries for bowlers
# ------------------------------------------

# Bowling-specific pressure definition:
# Required run rate must be between 8 and 15 (inclusive), AND no more
# than 24 legal balls must remain before the delivery.
bowling_df = batting_df[
    (batting_df["innings"] == 2)
    & (batting_df["required_run_rate_before_ball"] >= 8)
    & (batting_df["required_run_rate_before_ball"] <= 15)
    & (batting_df["balls_remaining_before_ball"] <= 24)
    & (batting_df["balls_remaining_before_ball"] > 0)
    & batting_df["bowler_id"].notna()
].copy()

# ------------------------------------------
# 12.2 Calculate runs conceded by each bowler
# ------------------------------------------

# Byes and leg-byes are not charged to the bowler. Penalty runs are
# also excluded. Wides and no-ball/overstep runs remain charged.
extras_type_clean = (
    bowling_df["extras_type"].fillna("").str.lower().str.strip()
)
non_bowler_extras = extras_type_clean.isin(["byes", "legbyes", "penalty"])

extras_total = (
    bowling_df["extras1_runs"].fillna(0)
    + bowling_df["extras_2_runs"].fillna(0)
)

bowling_df["bowler_runs_conceded"] = (
    bowling_df["total_runs"]
    - np.where(non_bowler_extras, extras_total, 0)
).clip(lower=0)

# ------------------------------------------
# 12.3 Identify wickets credited to the bowler
# ------------------------------------------

# These dismissal types are credited to the bowler in cricket.
# Run-outs, retired hurt/out, obstructing the field, and other
# non-bowler dismissals are deliberately excluded.
dismissal_type_clean = (
    bowling_df["dismissal_type"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace("_", " ", regex=False)
    .str.replace("-", " ", regex=False)
)

bowler_credited_dismissals = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket"
}

bowling_df["bowler_credited_wicket"] = (
    bowling_df["is_wicket"].fillna(False).astype(bool)
    & dismissal_type_clean.isin(bowler_credited_dismissals)
).astype(int)

# ------------------------------------------
# 12.4 Calculate over-wise pressure control
# ------------------------------------------

# Group only the bowler's high-pressure deliveries within each match over.
# The first pre-delivery RRR in that over represents the RRR at the
# beginning of the bowler's high-pressure over segment.
over_group_columns = [
    "bowler_id", "bowler_name", "match_id", "innings", "overid"
]

over_pressure = (
    bowling_df.sort_values("innings_ball_num")
    .groupby(over_group_columns)
    .agg(
        high_pressure_runs_conceded=("bowler_runs_conceded", "sum"),
        high_pressure_legal_balls=("legal_ball", "sum")
    )
    .reset_index()
)

# Get the actual RRR at the start of each over from ALL deliveries,
# not merely the first delivery that happened to meet the pressure filter.
pre_over_rrr = (
    batting_df.sort_values("innings_ball_num")
    .groupby(["match_id", "innings", "overid"])
    .agg(rrr_before_over=("required_run_rate_before_ball", "first"))
    .reset_index()
)

over_pressure = over_pressure.merge(
    pre_over_rrr,
    on=["match_id", "innings", "overid"],
    how="left",
    validate="many_to_one"
)

# Economy rate is undefined if the bowler has no legal high-pressure
# deliveries in that over (for example, only wides/no-balls).
over_pressure = over_pressure[
    over_pressure["high_pressure_legal_balls"] > 0
].copy()

over_pressure["high_pressure_economy"] = (
    over_pressure["high_pressure_runs_conceded"] * 6
    / over_pressure["high_pressure_legal_balls"]
)

over_pressure["over_pressure_control"] = (
    over_pressure["rrr_before_over"]
    - over_pressure["high_pressure_economy"]
)

# ------------------------------------------
# 12.5 Aggregate over-wise results by bowler
# ------------------------------------------

# No per-over scaling is applied. Each over's pressure-control value
# remains in its original units (runs per over).

# A match is counted for a bowler if they bowled at least one qualifying
# high-pressure delivery in that match. The average workload is calculated
# from legal high-pressure balls across those matches.
bowler_stats = (
    bowling_df
    .groupby(["bowler_id", "bowler_name"])
    .agg(
        high_pressure_deliveries=("bowler_id", "size"),
        high_pressure_legal_balls=("legal_ball", "sum"),
        high_pressure_runs_conceded=("bowler_runs_conceded", "sum"),
        high_pressure_wickets=("bowler_credited_wicket", "sum"),
        high_pressure_matches=("match_id", "nunique")
    )
    .reset_index()
)

over_control_stats = (
    over_pressure
    .groupby(["bowler_id", "bowler_name"])
    .agg(
        qualifying_high_pressure_overs=("over_pressure_control", "count"),
        average_over_pressure_control=("over_pressure_control", "mean"),
        total_pressure_control=("over_pressure_control", "sum"),
        average_required_run_rate=("rrr_before_over", "mean"),
        average_high_pressure_over_economy=("high_pressure_economy", "mean")
    )
    .reset_index()
)

bowler_stats = bowler_stats.merge(
    over_control_stats,
    on=["bowler_id", "bowler_name"],
    how="inner",
    validate="one_to_one"
)

bowler_stats["avg_high_pressure_balls_per_match"] = (
    bowler_stats["high_pressure_legal_balls"]
    / bowler_stats["high_pressure_matches"]
)

qualified_bowlers = bowler_stats[
    (bowler_stats["high_pressure_legal_balls"] >= MIN_HIGH_PRESSURE_BALLS_BOWLED)
    & (bowler_stats["high_pressure_matches"] >= MIN_HIGH_PRESSURE_MATCHES_BOWLED)
    & (bowler_stats["avg_high_pressure_balls_per_match"] > MIN_AVG_HIGH_PRESSURE_BALLS_PER_MATCH)
].copy()

# ------------------------------------------
# 12.6 Rank bowlers using pressure control + wickets
# ------------------------------------------

# Wicket rate is expressed as bowler-credited wickets per 6 legal
# high-pressure balls, so bowlers with different workloads can be compared.
qualified_bowlers["high_pressure_wicket_rate"] = np.where(
    qualified_bowlers["high_pressure_legal_balls"] > 0,
    qualified_bowlers["high_pressure_wickets"] * 6
    / qualified_bowlers["high_pressure_legal_balls"],
    0
)

# Both components are converted to percentile ranks because pressure
# control and wicket rate have different units and cannot be added directly.
PRESSURE_CONTROL_WEIGHT = 0.75
WICKET_WEIGHT = 0.25

if not qualified_bowlers.empty:
    qualified_bowlers["pressure_control_percentile"] = (
        qualified_bowlers["average_over_pressure_control"].rank(pct=True) * 100
    )

    qualified_bowlers["wicket_taking_percentile"] = (
        qualified_bowlers["high_pressure_wicket_rate"].rank(pct=True) * 100
    )

    qualified_bowlers["final_score"] = (
        PRESSURE_CONTROL_WEIGHT * qualified_bowlers["pressure_control_percentile"]
        + WICKET_WEIGHT * qualified_bowlers["wicket_taking_percentile"]
    )

    qualified_bowlers = qualified_bowlers.sort_values(
        by=["final_score", "pressure_control_percentile", "wicket_taking_percentile"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

# ------------------------------------------
# 12.6 Save and display the bowler ranking
# ------------------------------------------

bowler_ranking_file = os.path.join(
    RESULTS_PATH,
    "reliable_high_pressure_bowlers.csv"
)

qualified_bowlers.to_csv(bowler_ranking_file, index=False)

bowler_display_columns = [
    "bowler_id",
    "bowler_name",
    "high_pressure_legal_balls",
    "high_pressure_matches",
    "avg_high_pressure_balls_per_match",
    "qualifying_high_pressure_overs",
    "average_required_run_rate",
    "average_high_pressure_over_economy",
    "average_over_pressure_control",
    "total_pressure_control",
    "high_pressure_wickets",
    "high_pressure_wicket_rate",
    "pressure_control_percentile",
    "wicket_taking_percentile",
    "final_score"
]

print("Bowler pressure definition: 8 <= RRR <= 15 and 24 or fewer legal balls remaining")
print("Minimum high-pressure legal balls:", MIN_HIGH_PRESSURE_BALLS_BOWLED)
print("Minimum high-pressure matches:", MIN_HIGH_PRESSURE_MATCHES_BOWLED)
print("Minimum average high-pressure workload: greater than 0 legal balls per match")
print("Per-over pressure control: RRR at start of over - high-pressure economy")
print("Wicket rate: bowler-credited wickets per 6 legal high-pressure balls")
print("Final score: 75% pressure-control percentile + 25% wicket-taking percentile")
print("Qualified bowlers:", len(qualified_bowlers))

if not qualified_bowlers.empty:
    print("\n========== ALL QUALIFIED HIGH-PRESSURE BOWLERS ==========")
    print(qualified_bowlers[bowler_display_columns].to_string(index=False))
else:
    print("No bowlers meet the minimum high-pressure sample requirements.")

print("\nBowler ranking saved to:", bowler_ranking_file)

# ==========================================
# STEP 13: FINAL TOP 10 RANKINGS
# ==========================================

print("\n========== TOP 10 BATTERS ==========")

top_10_batters = (
    reliable_batters
    .sort_values("final_score", ascending=False)
    [["striker_name", "final_score"]]
    .head(10)
    .reset_index(drop=True)
)

top_10_batters.index += 1

print(top_10_batters.round(2).to_string())


print("\n========== TOP 10 BOWLERS ==========")

top_10_bowlers = (
    qualified_bowlers
    .sort_values("final_score", ascending=False)
    [["bowler_name", "final_score"]]
    .head(10)
    .reset_index(drop=True)
)

top_10_bowlers.index += 1

print(top_10_bowlers.round(2).to_string())
