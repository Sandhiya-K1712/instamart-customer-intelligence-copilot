# Instamart Customer Intelligence & Support Copilot

## Overview

This project develops an offline customer intelligence and support
copilot for an Instamart-style e-commerce dataset.

The project covers four major areas:

-   **Part A:** Data auditing, cleaning, SQLite database creation, SQL
    queries and exploratory data analysis.

-   **Part B:** RFM analysis, customer segmentation, customer lapse
    prediction and retention targeting.

-   **Part C:** Customer support ticket classification, PII redaction,
    prompt comparison, token budgeting and ticket routing.

-   **Part D:** Policy document chunking, TF-IDF retrieval, policy
    question answering and a tool-using ReAct refund agent with safety
    limits and checkpoint recovery.

All language-model interactions use the supplied `mock_llm.py`. No paid
or external LLM API is required.

## Reproduce

### Requirements

-   Python 3

-   Git

-   Packages listed in `requirements.txt`

### Setup

Clone the repository and navigate to the project directory.

``` bash

git clone https://github.com/Sandhiya-K1712/instamart-customer-intelligence-copilot.git

cd instamart-customer-intelligence-copilot
```

Create and activate a virtual environment.

Windows PowerShell:

``` powershell

python -m venv .venv

.venv\Scripts\Activate.ps1
```

Install dependencies:

``` bash

pip install -r requirements.txt
```

### Generate data

Run the supplied data generator:

``` bash

python generate_data.py
```

The generator produces 1,200 customers, 40 products, 13,084 raw orders
and 164 support tickets, along with policy and agent scenario files.

### Run the project

Run the scripts in the following order:

``` bash

python src/audit_data.py

python src/clean_data.py

python src/build_database.py

python src/validate_data.py

python src/run_sql.py

python src/eda_analysis.py

python src/part_b_analysis.py

python src/part_c_triage.py

python src/part_d_agent.py
```

The scripts generate the cleaned datasets, database, analysis outputs,
trained model, ticket triage results, agent execution logs and
checkpoint files.

## Part A

### Data audit

Audited the raw customer, product, order and ticket datasets.

Key findings:

| Dataset \| Rows \| Important findings \|

\|---\|---:\|---\|

| Customers \| 1,200 \| 61 missing ages, 30 missing genders \|

| Products \| 40 \| No missing values \|

| Orders \| 13,084 \| 119 exact duplicates, 120 nonnumeric amounts, 106
  negative delivery values \|

| Tickets \| 164 \| 4 empty ticket texts \|

Additional findings included 2,541 missing delivery-time values, 656
pending orders and 24 raw city spellings.

### Data cleaning

-   Standardised customer cities and gender values.

-   Handled missing and invalid customer ages.

-   Removed exact duplicate orders.

-   Removed duplicate order IDs while retaining the latest order record.

-   Converted order amounts and timestamps to appropriate data types.

-   Recalculated invalid order amounts using product price, quantity and
    discount.

-   Replaced negative delivery times with missing values.

-   Filled missing delivery times for completed orders using category
    medians.

Final cleaned orders: **12,309**.

| Order status \| Count \|

\|---\|---:\|

| Completed \| 11,110 \|

| Cancelled \| 1,199 \|

| Pending \| 0 \|

Completed order revenue: **INR 2,998,668.80**.

### Database and SQL

Created `data/instamart.db` with customers, products and orders tables.

Implemented seven SQL queries covering:

-   Top customers by spending.

-   Revenue by city and product category.

-   Products priced above the average.

-   Customers without completed orders.

-   Customers spending above average.

-   Top three customers in each city.

-   Daily revenue and cumulative revenue.

All seven queries executed successfully.

The final cumulative revenue was INR 2,998,668.80.

### Exploratory data analysis

-   Average completed order amount: INR 269.91.

-   Median completed order amount: INR 135.00.

-   Standard deviation and distribution analysis identified a
    right-skewed order-value distribution.

-   Skewness: 4.61.

-   Interquartile range: INR 232.20.

-   Number of high-value outliers above the upper fence: 897.

-   Correlation between order amount and discount percentage: -0.095.

-   Correlation between order amount and quantity: 0.087.

## Part B

### RFM analysis

Performed Recency, Frequency and Monetary analysis using completed
orders.

-   Customers included: 1,175.

-   Average purchase frequency: 9.46.

-   Average monetary value: INR 2,552.06.

-   Median recency: 19 days.

### Customer segmentation

Used K-Means clustering and evaluated cluster counts from K=2 to K=8.

Selected **K=4** and assigned the following segment names:

1\. Occasional Shoppers

2\. Premium Customers

3\. Frequent Loyalists

4\. Dormant Customers

The Premium Customers segment contained 138 customers and had an average
monetary value of INR 7,129.70.

The selected K=4 solution achieved a silhouette score of 0.526. PCA on
the standardised RFM data explained 63.4% of the variance with the first
component and 21.8% with the second component, for 85.2% total explained
variance.

### Customer lapse prediction

Defined the lapse cohort using a cutoff date of 2025-09-01 and a label
period ending 2025-10-01.

-   Cohort size: 1,156.

-   Lapsed customers: 443.

-   Lapse rate: 38.32%.

Compared Logistic Regression and Random Forest models using
training-only preprocessing and cross-validation.

| Model \| Test precision \| Test recall \| Test F1 \|

\|---\|---:\|---:\|---:\|

| Logistic Regression \| 0.6444 \| 0.6517 \| 0.6480 \|

| Random Forest \| 0.6404 \| 0.6404 \| 0.6404 \|

| Logistic Regression with training-only oversampling \| 0.5891 \|
  0.8539 \| 0.6972 \|

The oversampled Logistic Regression model improved recall from 65.17% to
85.39%.

Missed lapsed customers decreased from 31 to 13, while false positives
increased from 32 to 53.

The trained model was saved and reloaded successfully. Predictions
matched after reloading.

The Random Forest model achieved the best cross-validation F1 score of
0.6865. The final retention list selected 10 Premium customers; 2 of
these customers were actually lapsed.

Generated a retention targeting list in `outputs/retention_list.csv`.

## Part C

### Ticket classification and PII redaction

Processed 164 support tickets.

Redacted personal information before sending ticket text to the mock
language model.

| PII type \| Substitutions \|

\|---\|---:\|

| Email addresses \| 55 \|

| Phone numbers \| 91 \|

| Order IDs \| 132 \|

| Total substitutions \| 278 \|

Verified zero remaining email addresses, phone numbers and order IDs
after redaction.

Input guardrails:

-   Empty tickets: 4.

-   Long tickets: 4.

-   Empty and long tickets were skipped rather than sent to the model.

### Zero-shot and few-shot comparison

Compared zero-shot and few-shot classification across four categories:
billing, delivery, fraud and technical.

| Metric \| Zero-shot \| Few-shot \|

\|---\|---:\|---:\|

| Classified tickets \| 147 \| 147 \|

| Correct classifications \| 126 \| 147 \|

| Accuracy \| 85.71% \| 100% \|

| Total tokens \| 11,116 \| 19,696 \|

| Tokens per classified ticket \| 75.62 \| 133.99 \|

Few-shot prompting improved accuracy by 14.29 percentage points, with an
additional 8,580 tokens.

### Retry handling and budget demonstration

-   Used a maximum of three retries with exponential backoff and jitter.

-   Recovered from 19 transient errors.

-   Validated model output and used keyword fallback when necessary.

-   Used a 6,000-token budget for the budget demonstration.

Budget demonstration:

-   Total tokens: 6,010.

-   Budget-exhausted tickets: 72.

-   Classified tickets: 80.

-   Correct classifications: 67.

-   Accuracy among classified tickets: 83.75%.

The final few-shot run used 19,696 tokens and classified all 147
eligible tickets.

### Ticket routing

Final routing results:

| Route \| Tickets \|

\|---\|---:\|

| support_queue \| 102 \|

| security_team \| 34 \|

| human_review \| 17 \|

| premium_desk \| 11 \|

Generated `outputs/triage_results.csv` and
`outputs/triage_budget_demo.csv`.

## Part D

### Policy document chunking and retrieval

Processed five policy documents using 40-word chunks with 10-word
overlap.

-   Total chunks: 15.

-   Delivery policy: 3.

-   Escalation policy: 3.

-   Payment policy: 3.

-   Privacy policy: 2.

-   Refund policy: 4.

Implemented TF-IDF vectorisation and cosine similarity retrieval.

The retrieval test for "refund policy late delivery" returned three
refund-policy chunks.

### Policy assistant evaluation

Evaluated 14 questions: 12 answerable and 2 unanswerable.

| Evaluation \| Result \|

\|---\|---:\|

| Expected document retrieved in top 3 \| 12/12 \|

| Expected fact found in answer \| 12/12 \|

| Correct unanswerable fallbacks \| 2/2 \|

Compared with zero-overlap chunking:

-   Zero-overlap chunks: 13.

-   Answerable questions tested: 12.

-   Expected facts missing from the retrieved top 3: 1.

The missed fact concerned the 48-hour payment reversal policy,
illustrating how chunk boundaries can affect retrieval.

### Refund tools and ReAct agent

Implemented five tools:

-   `get_order`

-   `search_policy`

-   `calculate_refund`

-   `issue_refund`

-   `escalate_to_human`

Refund rules:

-   Only completed orders qualify.

-   Delivery exceeding 45 minutes qualifies for a full order-amount
    refund.

-   Delivery within 45 minutes does not qualify.

-   Refunds above INR 1,000 require human approval.

The agent uses an iterative ReAct loop to inspect order details,
retrieve policies, calculate refunds and either issue or escalate the
request.

### Scenario results

| Scenario \| Status \| Refund amount (INR) \| Model calls \| Tokens \|

\|---\|---\|---:\|---:\|---:\|

| S1 \| REFUND \| 128.25 \| 5 \| 1,127 \|

| S2 \| REFUND \| 135.00 \| 5 \| 1,127 \|

| S3 \| ESCALATE \| 1,999.00 \| 5 \| 1,114 \|

| S4 \| DENIED \| 0.00 \| 4 \| 785 \|

| S5 \| DENIED \| 0.00 \| 3 \| 472 \|

| S6 \| NOT_FOUND \| 0.00 \| 2 \| 163 \|

| S7 \| FALLBACK_MAX_ITERATIONS \| Not issued \| 6 \| 3,270 \|

### Safety limits

Tested three safeguards:

-   Maximum iteration limit: S7 returned `FALLBACK_MAX_ITERATIONS`.

-   Token limit: a 200-token limit triggered `FALLBACK_TOKEN_LIMIT`
    after 3 model calls and 489 reported tokens.

-   Time limit: a zero-second limit triggered `FALLBACK_TIMEOUT` before
    the first model call.

### Checkpoint and resume

Implemented JSON checkpoint persistence after every tool observation.

Simulated a crash after the second tool execution in S1.

-   Tools completed before crash: `get_order`, `search_policy`.

-   Restored the conversation and execution state from
    `state/S1_CRASH_DEMO.json`.

-   Resumed with `calculate_refund` and `issue_refund`.

-   Final status: REFUND.

-   Final refund: INR 128.25.

-   Total model calls across the original and resumed execution: 5.

-   Total tokens: 1,127.

The completed tools were not repeated after resuming.

### ReAct vs Plan-and-Execute

The ReAct approach combines reasoning and tool execution in an iterative
loop. The agent observes each tool result and decides its next action
based on the returned information. In this project, the ReAct agent used
2--6 model calls across the seven scenarios. Successful refund scenarios
generally required 5 calls, while a missing order required only 2 calls.

A Plan-and-Execute approach could reduce repeated reasoning by creating
a plan before executing tools. However, a fixed plan may be less
flexible when unexpected observations occur, such as a missing order, a
cancelled order or a refund requiring human approval.

ReAct is more suitable for this refund workflow because decisions depend
on actual database results and policy checks. Although it may consume
more tokens, its adaptability and checkpoint-based recovery make it the
preferred approach.

## Limitations and Next Steps

-   The dataset is synthetic and may not represent real customer
    behaviour.

-   TF-IDF retrieval relies on word overlap and may not understand
    semantic paraphrases.

-   The mock language model does not represent the performance of a
    production LLM.

-   The lapse prediction model is trained and evaluated on a limited
    synthetic cohort.

-   The retention list is illustrative; its Premium segment flag uses
    the full RFM dataset, including September activity, so it is not a
    leakage-free evaluation.

-   The refund ledger records simulated actions only. No real payments
    or refunds are processed.

-   A production implementation would require authentication, access
    controls, secure PII handling, monitoring, human approval workflows
    and integration with real order and payment systems.

Future improvements could include semantic retrieval, broader model
validation, monitoring for model drift, more extensive agent testing and
secure integration with production support systems.
