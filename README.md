### MMM Baseline Model

### Project Overview:

A Marketing Mix Model answers one simple question:

Of all the revenue a company made, how much of it actually came from advertising and which channel deserves the credit?

Imagine a company spends money on TV ads, radio ads, and Google search ads every week. At the end of the year, they made €10 million. But how much of that €10 million happened because of the TV ads? How much because of search? And how much would have happened anyway, even if they had spent zero on advertising — because the brand already has loyal customers, or because people just needed the product regardless?
That is exactly what this model estimates. It splits every euro of revenue into two buckets:

Media-driven revenue: revenue that was caused by advertising
Baseline revenue: revenue that would have happened with zero ad spend

### The Data
The dataset comes from a publicly available simulated marketing dataset (source: Datankist). It contains 208 rows, one per week, covering roughly 4 years from 2016 to 2019.
Think of it as a spreadsheet where each row is one week. For every week, we know how much was spent on each ad channel and how much revenue came in.
### Columns
ColumnTypeWhat it meansDATEDateThe Monday of each weektv_SNumericEuros spent on TV advertising that weekradio_SNumericEuros spent on radio advertising that weekpaid_search_SNumericEuros spent on paid search ads that weekrevenueNumericTotal company revenue that week — the variable the model is trying to explaincompetitor_salesNumericA competitor's sales figure — excluded from all models (see below)
Why is competitor_sales excluded?
Before fitting any model, a data quality check measures the correlation between every variable and revenue. Correlation is a number between -1 and +1 that tells you how closely two variables move together. A correlation of 1.0 means they are perfectly in sync — when one goes up, the other always goes up by the exact same proportion.
competitor_sales had a correlation of 1.0 with revenue in this dataset. This is a data artefact — they are essentially the same number. Including it would give the model a perfect R² of 1.0 by simply using competitor_sales as a stand-in for revenue, completely ignoring the actual ad channels. It tells us nothing useful and breaks the model entirely. It is permanently excluded from all features.

### Key concepts: definitions
Before explaining what the model does, here are plain-English definitions of every concept used.
### Regression
A statistical method that finds the relationship between one outcome variable (revenue) and one or more input variables (ad spend, seasonality, trend). It answers: "for every extra unit of TV spend, how many extra euros of revenue do we typically see?" The output is a set of coefficients — one per input variable.
### Coefficient
A number the model assigns to each variable. If the TV coefficient is 2.5, it means: for every one unit increase in transformed TV spend, the model estimates €2.50 of additional revenue. Larger coefficient = stronger estimated relationship between that channel and revenue.
### R² (R-squared)
A number between 0 and 1 that tells you how much of the week-to-week variation in revenue the model explains. An R² of 0.87 means the model explains 87% of why revenue goes up and down each week. The remaining 13% is variation the model cannot capture.
### MAPE (Mean Absolute Percentage Error)
A measure of prediction accuracy. If MAPE = 10%, it means on average the model's revenue predictions are 10% away from the actual revenue. Lower is better. Under 10% is generally considered good for a first marketing mix model.
### Multicollinearity
This happens when two or more input variables are highly correlated with each other. For example, if a company always runs TV and radio ads at the same time, it becomes mathematically very difficult to separate their individual effects. The model cannot tell if a revenue spike was caused by TV, radio, or both. High multicollinearity makes coefficients unstable and untrustworthy.
### VIF (Variance Inflation Factor)
A number that measures how severe the multicollinearity is for each variable:

VIF below 4 — fine, no problem
VIF 4 to 10 — moderate, worth monitoring
VIF above 10 — serious, coefficients are unreliable

### Regularisation
A technique used in Ridge regression that adds a small penalty to prevent any one variable from receiving an unrealistically large coefficient. It deliberately trades a tiny bit of accuracy for a lot of stability — useful when channels are correlated.
### Baseline revenue
The revenue the company would have made even with zero advertising. This includes loyal repeat customers, organic traffic, word of mouth, and general market demand. In most real MMMs, baseline is 60–80% of total revenue — advertising is incremental on top of an existing business.

### What transformations are applied to the data?
Raw spend numbers cannot go straight into a regression model. Two transformations are applied first, in this exact order: adstock first, then saturation.

### Transformation 1: Adstock
What is adstock and why is it needed?
When someone sees a TV ad on Monday, they might not buy the product until Friday. When a brand runs a big campaign in January, people may still remember it in March. The advertising effect lingers for weeks after the ad ran.
This carry-over effect is called adstock. Without it, the model only credits the week the ad ran and misses all the weeks it continued working — so it underestimates the true effect of advertising.
### What type of adstock is used?
This model uses geometric adstock, also called exponential decay adstock. It is the simplest, most widely used adstock formulation in MMM.
The formula is:
adstock_t = spend_t + decay × adstock_{t-1}
This week's adstock = this week's actual spend + a fraction of last week's adstock. That fraction is the decay parameter.
### What is the decay parameter?
A number between 0 and 1 that controls how quickly the advertising effect fades:

Decay = 0 → effect disappears instantly, nothing carries over
Decay = 1 → effect never fades (unrealistic)
Decay = 0.5 → each week, 50% of last week's effect remains

### How were the decay values chosen?
Set based on industry convention and domain knowledge, which is standard practice when no prior model runs exist to calibrate from:
ChannelDecayReasoningTV0.6TV builds brand memory over weeks — long carry-overRadio0.3Medium carry-over — more immediate than TVPaid Search0.1Almost entirely immediate — you click or you don't
### What does geometric adstock look like in practice?
If €1,000 is spent on TV in week 1 and nothing after:
Week 1:  1000
Week 2:  0 + 0.6 × 1000 = 600
Week 3:  0 + 0.6 × 600  = 360
Week 4:  0 + 0.6 × 360  = 216
...fades gradually to zero
The effect does not stop abruptly — it decays smoothly, which reflects how memory and brand awareness actually work.

### Transformation 2: Saturation
What is saturation and why is it needed?
Doubling your ad spend does not double your sales. The first €1,000 spent on TV reaches a large fresh audience. The next €1,000 reaches people who already saw the ad. The next €1,000 reaches even more of the same people. At some point, almost everyone reachable has been reached — additional spend produces almost no additional revenue.
This is called diminishing returns. Without modelling it, a linear regression assumes spend and revenue have a straight-line relationship — spend double, get double. That is unrealistic and produces inflated channel contribution estimates.
### What type of saturation is used?
This model uses the Michaelis-Menten function, one of the most common saturation curves in MMM.
The formula is:
S(x) = x / (x + K)
Where:

x is the adstock-transformed spend value for that week
K is the half-saturation constant — the spend level at which 50% of the maximum possible effect is reached
Output is always between 0 and 1

### What does this curve look like?

At very low spend: the curve rises steeply — each extra euro has a large effect
At spend = K: you are at exactly 50% of maximum possible effect
At very high spend: the curve flattens — each extra euro adds almost nothing
The output never reaches 1.0 — there is always theoretically more room, but in practice you get very close at high spend levels

This is a concave curve, diminishing returns from the very first euro of spend.
Is this the same as logistic saturation?
No. They are different shapes:

Michaelis-Menten (used here): diminishing returns from the very first euro. The curve only ever decelerates. No threshold needed before the effect kicks in.
Logistic: an S-shape — starts flat, accelerates through a middle zone, then decelerates. Models a threshold effect where spend has little impact until a critical mass is reached.

Michaelis-Menten is the standard default in MMM because most channels show diminishing returns from the start. The logistic shape implies there is a minimum spend level below which advertising does almost nothing — which is harder to justify for general brand advertising.
### How was K chosen?
K is set automatically to the median of non-zero adstock values for each channel:
pythonK = median of all weeks where adstock > 0
This is a data-driven default. It places the half-saturation point at the middle of the observed spend distribution, roughly half of all observed spend weeks are below the point of maximum efficiency, and half are above it. No manual tuning is required.
Why is saturation applied after adstock, not before?
Order matters. Adstock is applied first because carry-over happens at the media level — impressions and exposures accumulate over time. Once the total effective exposure has been calculated (adstock), saturation is applied to model how that accumulated exposure translates into consumer response. Applying saturation before adstock would model diminishing returns on single-week spend before accounting for carry-over, which is economically incorrect.

### Seasonality and trend controls
Why are seasonality controls needed?
Revenue is not flat across the year. December is almost always higher than February. If the model does not control for these patterns, it will mistakenly credit advertising for revenue that was always going to happen due to the season.
### What type of seasonality is used?
This model uses monthly dummy variables. A dummy variable is a binary (0 or 1) variable that simply flags whether a given week falls in a particular month.
Eleven dummies are created — one for each month from February to December — with January as the reference category (dropped to avoid perfect multicollinearity). Each dummy captures the average revenue effect of being in that calendar month, independent of advertising spend.
For example, a December dummy coefficient of +50,000 means: on average, December weeks generate €50,000 more revenue than January weeks, holding advertising constant.
Monthly dummies were chosen over Fourier terms (sine and cosine waves) for two reasons:

It is simpler to explain because a December dummy is immediately interpretable; a sine wave is not
It is appropriate for the data — with 208 weeks, 11 monthly dummies is well within acceptable parameter count

### What is the trend variable?
A simple counter from 0 to 207, one per week. It captures long-run growth or decline in revenue that is unrelated to advertising or seasonality.
For example, if the company was growing steadily over 4 years, the trend variable absorbs that upward drift so it is not mistakenly credited to an advertising channel.

### The models
Both models use exactly the same features:

tv_sat, radio_sat, search_sat (adstock + Michaelis-Menten saturation applied)
trend
11 monthly dummy variables

The only difference between them is the estimation method.
### Model 1: OLS (Ordinary Least Squares)
The standard regression method. Finds the set of coefficients that minimises the sum of squared differences between predicted and actual revenue. No restrictions, no penalties.
Strength: Simple, interpretable, standard in academic work.
Weakness: Unstable when input variables are correlated. When TV and radio tend to run together, OLS struggles to correctly separate their individual contributions — it may assign too much credit to one and too little to the other.
### Model 2: Ridge Regression
OLS with one addition: a regularisation penalty. The model minimises:
sum of squared errors  +  α × sum of squared coefficients
The penalty term discourages any single channel from receiving an unrealistically large coefficient. This makes estimates more stable when channels are correlated, because instead of letting one coefficient explode to compensate for another, Ridge shrinks all coefficients proportionally toward zero.
Strength: More stable than OLS under multicollinearity. Better generalisation on unseen data in many real-world datasets.
Weakness: Coefficients are intentionally biased (shrunk), so they are not pure unbiased estimates.

### How is alpha chosen?
Alpha controls the strength of the penalty. RidgeCV is used, which automatically selects the best alpha from the candidate list [0.01, 0.1, 1, 10, 50, 100, 500] using 5-fold cross-validation. This means the data itself determines how much regularisation is appropriate — no arbitrary manual choice is needed.
5-fold cross-validation means: the training data is split into 5 equal chunks. The model is trained on 4 chunks and tested on the 5th, rotating through all combinations. The alpha that produces the best average performance across all 5 rotations is selected.

### Outputs
Channel contributions
For each channel, the model multiplies its coefficient by the channel's transformed spend values across all 208 weeks and sums the result. This gives the total revenue attributed to that channel over 4 years, expressed as a percentage of total revenue. Negative contributions are floored at zero. The remainder is the baseline.

### Out-of-sample validation (80/20 split)
The model is trained on the first 80% of weeks (weeks 1–166) and asked to predict the final 20% (weeks 167–208) it has never seen. MAPE on this held-out period checks whether the model is learning a genuine relationship or memorising the training data.
Both OLS and Ridge are validated this way. Ridge uses fit_transform on the training set and transform (not fit_transform) on the test set — this is important because the scaler must be fitted only on training data, not the test data, to simulate a genuine out-of-sample prediction.
Plots
The script produces a 2-panel chart saved to baseline_results.png:

Left panel: Actual vs OLS fitted revenue across all 208 weeks
Right panel: Channel contribution percentages for OLS vs Ridge side by side


### Why two models?
Comparing OLS and Ridge is the foundation of the thesis research question:

How does inter-channel multicollinearity affect the stability of channel contribution estimates in a Marketing Mix Model and does a Bayesian framework with informative priors produce more robust estimates than OLS and Ridge under high channel correlation conditions?

The next stage builds a Bayesian MMM with informative priors as a third estimator alongside OLS and Ridge. 
Once all three models are built and compared on the real dataset, the experiment moves to synthetic data with artificially controlled correlation levels (0.0 and 0.7) to test stability under high multicollinearity conditions.
