# Team Meeting Agenda - Preprocessing Module Integration

## Meeting Objectives

1. **Align on data contracts** between Analysis and Preprocessing teams
2. **Clarify responsibilities** for each team member
3. **Set timeline** for implementation and integration
4. **Resolve blockers** (especially missing input format from Analysis)

---

## 1. Project Overview (5 min)

**Current Status:**
- ✅ Project structure established
- ✅ Mock data framework ready for parallel development
- ✅ Yang's numerical preprocessing functions complete
- ⏳ Awaiting analysis module output format from Adrian & Alisa
- ⏳ Amruth & Tecla functions pending

**System Workflow:**
```
Dataset → analyze.py → preprocessing_suggestions.py → main.py (user approval) → preprocessing_function.py → Output
```

---

## 2. Critical Blocker: Data Contract (15 min)

### ⚠️ Problem
Our preprocessing module needs the **exact output format** from Adrian & Alisa's analysis module to finalize implementation.

### 🎯 What We Need from Adrian & Alisa

**Question:** What will your `analyze.py` return?

We've created a **proposed structure** in `DATA_CONTRACT.md`. Please review:

```python
analysis_report = {
    'feature_types': {'Age': 'numerical', 'Sex': 'categorical', ...},
    'missing_values': {'Age': {'count': 177, 'percentage': 0.19, ...}, ...},
    'outliers': {'Fare': {'has_outliers': True, 'count': 23, ...}, ...},
    'numerical_stats': {'Age': {'skewness': 0.38, 'mean': 29.7, ...}, ...},
    'categorical_stats': {'Sex': {'cardinality': 2, ...}, ...},
    'duplicates': {'has_duplicates': False, 'count': 0},
    ...
}
```

**Action Items:**
- [ ] Adrian & Alisa: Review and confirm/modify this structure
- [ ] Adrian & Alisa: Provide example output (even if analysis.py not complete)
- [ ] Adrian & Alisa: Share timeline for completion

---

## 3. Parallel Development Strategy (10 min)

### Proposal

**Option A: Use Mock Data (Recommended)**
- We've created `mock_analysis_output.py` with realistic test data
- Teams can work in parallel without blocking each other
- Easy to adjust later when actual analyze.py is ready

**Option B: Wait for Analysis Module**
- Preprocessing team waits for Adrian & Alisa to finish
- Risk: Delays our implementation

**Option C: Hybrid Approach**
- Adrian & Alisa provide structure NOW (10 min discussion)
- Share example even if code incomplete
- We proceed with implementation

### Decision
**Vote:** Which approach should we take?

---

## 4. Team Responsibilities (10 min)

### Yang/Zhiqi ✅ Status: Complete

**Completed:**
- `standard_scaler()` - Z-score normalization
- `minmax_scaler()` - Scale to [0,1] range  
- `clip_outliers_iqr()` - Cap outliers to boundaries
- `remove_outliers_iqr()` - Delete rows with outliers

**Remaining:**
- Implement rule logic in `preprocessing_suggestions.py`:
  - `_suggest_numerical_scaling()`
  - `_suggest_outlier_handling()`

**Timeline:** Ready for integration testing

---

### Amruth ⏳ Status: Pending

**To Implement:**
1. **Preprocessing Functions** (`preprocessing_function.py`):
   - `encode_one_hot()` - One-hot encoding
   - `encode_label()` - Label encoding  
   - `encode_target()` - Target encoding

2. **Rule Logic** (`preprocessing_suggestions.py`):
   - `_suggest_categorical_encoding()`

**Rules to Implement:**
```python
if cardinality == 2:
    → use label encoding
elif cardinality <= 10:
    → use one-hot encoding
else:
    → use target encoding
```

**Timeline:** When can this be completed?

---

### Tecla ⏳ Status: Pending

**To Implement:**
1. **Preprocessing Functions** (`preprocessing_function.py`):
   - `impute_median()` - Fill missing with median
   - `impute_mean()` - Fill missing with mean
   - `impute_mode()` - Fill missing with mode
   - `remove_duplicates()` - Delete duplicate rows
   - `delete_missing_rows()` - Delete rows with missing values

2. **Rule Logic** (`preprocessing_suggestions.py`):
   - `_suggest_missing_values()`
   - `_suggest_duplicates()`

**Rules to Implement:**
```python
if missing_rate < 5%:
    → delete rows
elif numerical:
    → median imputation
elif categorical:
    → mode imputation
```

**Timeline:** When can this be completed?

---

### Adrian & Alisa ⏳ Status: In Progress

**To Complete:**
- `analyze.py` - Analysis module
- Confirm output format (data contract)
- Share example output for testing

**Timeline:** When will this be ready?

---

## 5. Files to Review Together (10 min)

### Key Files Created

1. **`DATA_CONTRACT.md`** ⭐ MOST IMPORTANT
   - Defines interfaces between modules
   - **Action:** Everyone review and approve

2. **`mock_analysis_output.py`**
   - Example of analyze.py output
   - Use for parallel development
   - **Action:** Adrian & Alisa verify accuracy

3. **`main.py`**
   - Integration pipeline
   - Demonstrates end-to-end workflow
   - **Action:** Everyone understand the flow

4. **`preprocessing_suggestions.py`**
   - Rule engine framework
   - Each person has their section
   - **Action:** Review your section

5. **`SETUP_GUIDE.md`**
   - Installation instructions
   - Usage examples
   - **Action:** Everyone follow setup

---

## 6. Timeline & Milestones (10 min)

### Proposed Timeline

**Week 1: Now → Friday**
- [ ] Confirm data contract (Adrian & Alisa)
- [ ] Amruth implements categorical functions
- [ ] Tecla implements missing value functions
- [ ] Yang finalizes rule logic

**Week 2: Next Week**
- [ ] Integration testing with mock data
- [ ] Connect with actual analyze.py (if ready)
- [ ] Bug fixes and adjustments

**Week 3: Following Week**
- [ ] End-to-end testing
- [ ] Documentation
- [ ] Final integration

### Questions
- Is this timeline realistic?
- Any blockers or concerns?
- Do we need to adjust?

---

## 7. Testing Strategy (5 min)

### Unit Testing
- Each person tests their own functions
- Use Titanic dataset (or similar)
- Verify functions work independently

### Integration Testing
- Test with `main.py` in demo mode
- Use mock data until analyze.py ready
- Identify and fix interface issues

### End-to-End Testing
- Full pipeline with real dataset
- Verify user experience
- Performance testing

---

## 8. Communication Plan (5 min)

### Daily Updates
- **Where:** Team chat/Slack
- **When:** End of day
- **What:** Progress, blockers, questions

### Code Review
- **How:** Pull requests
- **Who:** At least one other team member
- **When:** Before merging to main

### Documentation
- **Where:** Update `DATA_CONTRACT.md` when changes occur
- **Who:** Person making the change
- **When:** Immediately

---

## 9. Open Discussion & Q&A (10 min)

### Questions for the Team

1. Do you understand your responsibilities?
2. Do you have the information you need to start?
3. Are there any technical questions?
4. Any concerns about the timeline?
5. How can we help unblock each other?

---

## 10. Action Items Summary

### Adrian & Alisa
- [ ] Review `DATA_CONTRACT.md` and confirm analysis output structure
- [ ] Provide example output (even if mock/incomplete)
- [ ] Share timeline for analyze.py completion
- [ ] Deadline: [Set date]

### Yang/Zhiqi
- [x] Complete numerical preprocessing functions ✅
- [ ] Implement rule logic in preprocessing_suggestions.py
- [ ] Test integration with mock data
- [ ] Deadline: [Set date]

### Amruth
- [ ] Implement categorical encoding functions
- [ ] Implement rule logic in preprocessing_suggestions.py
- [ ] Test with categorical data examples
- [ ] Deadline: [Set date]

### Tecla
- [ ] Implement missing value handling functions
- [ ] Implement duplicate handling functions
- [ ] Implement rule logic in preprocessing_suggestions.py
- [ ] Deadline: [Set date]

### Everyone
- [ ] Review `DATA_CONTRACT.md`
- [ ] Set up development environment (pandas, numpy)
- [ ] Test `main.py` in demo mode
- [ ] Communicate blockers immediately

---

## Quick Demo (if time permits)

**Run the current framework:**
```bash
cd European-Masters-Team-Project
python3 main.py
```

Shows:
- Mock analysis data
- Generated suggestions
- Expected workflow

---

## Meeting Notes

**Date:** _______________

**Attendees:**
- [ ] Yang/Zhiqi
- [ ] Amruth
- [ ] Tecla
- [ ] Adrian
- [ ] Alisa

**Decisions Made:**

1. Data contract approach: _______________
2. Timeline agreed: _______________
3. Next meeting: _______________

**Parking Lot (topics for later):**
-
-

---

**Next Meeting:** [Date/Time]  
**Goal:** Integration testing results and final adjustments

