import os, sys, re, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_auc_score)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.calibration import CalibratedClassifierCV

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

warnings.filterwarnings('ignore')

print("=" * 65)
print("  MedicalAI Pro — NLP Model Training")
print("  Clinical Text Classification (Medical Transcriptions)")
print("=" * 65)
print()


print("STEP 0: Downloading NLTK resources...")
for pkg in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
print("  NLTK resources ready.\n")


print("STEP 1: Loading Kaggle Dataset (mtsamples.csv)...")
print("-" * 50)

DATASET_PATH = "mtsamples.csv"

if not os.path.exists(DATASET_PATH):
    print(f"  ❌  '{DATASET_PATH}' NOT FOUND in current directory.")
    print()
    print("  ➜  Download from Kaggle:")
    print("     https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions")
    print("     Place 'mtsamples.csv' in the MedicalAI_Pro folder, then re-run.")
    print()
    print("  ➜  Running in DEMO mode with a built-in small sample so you can")
    print("     test the code structure before getting the full dataset.\n")
    DEMO_MODE = True
else:
    DEMO_MODE = False

if DEMO_MODE:
    
    demo_records = [
        ("Cardiology",      "Patient presents with chest pain, shortness of breath, elevated troponin. ECG shows ST elevation. Diagnosis: acute myocardial infarction."),
        ("Cardiology",      "Hypertensive urgency. BP 190/110. Started on amlodipine. Echocardiogram shows left ventricular hypertrophy."),
        ("Cardiology",      "Atrial fibrillation with rapid ventricular response. Cardioversion performed. Warfarin anticoagulation initiated."),
        ("Cardiology",      "Congestive heart failure exacerbation. Bilateral crackles. BNP elevated. Started on IV furosemide."),
        ("Cardiology",      "Stable angina. Nuclear stress test positive. Referred for cardiac catheterization and possible stenting."),
        ("Neurology",       "Sudden onset right-sided weakness and aphasia. MRI diffusion shows acute ischemic stroke in left MCA territory."),
        ("Neurology",       "Migraine with aura. Visual scotoma preceding headache. Treated with sumatriptan. Prophylaxis with topiramate."),
        ("Neurology",       "Epilepsy follow-up. Generalized tonic-clonic seizures controlled on levetiracetam. EEG shows interictal discharges."),
        ("Neurology",       "Parkinson disease. Bradykinesia, rigidity, resting tremor. Started on carbidopa-levodopa. Physical therapy referral."),
        ("Neurology",       "Multiple sclerosis relapse. New MRI lesions. Treated with IV methylprednisolone. Natalizumab therapy discussed."),
        ("Orthopedics",     "Right knee osteoarthritis grade III. Medial compartment narrowing. Cortisone injection given. Total knee replacement planned."),
        ("Orthopedics",     "Lumbar disc herniation L4-L5. Radiculopathy down the right leg. MRI confirms disc extrusion. Physical therapy initiated."),
        ("Orthopedics",     "Proximal humerus fracture. ORIF performed. Post-op X-ray shows good alignment. Sling for 6 weeks."),
        ("Orthopedics",     "Anterior cruciate ligament tear. ACL reconstruction with patellar tendon graft. Rehab protocol 9 months."),
        ("Orthopedics",     "Cervical spondylosis with myelopathy. Anterior cervical discectomy and fusion C5-C6 performed."),
        ("Gastroenterology","Colonoscopy reveals three tubular adenomas removed. Pathology benign. Repeat colonoscopy in 3 years."),
        ("Gastroenterology","Crohn disease flare. Elevated CRP and fecal calprotectin. Steroids tapered; infliximab maintenance continued."),
        ("Gastroenterology","Upper GI bleed secondary to peptic ulcer. Endoscopic hemostasis achieved. H. pylori eradication therapy started."),
        ("Gastroenterology","Chronic hepatitis C genotype 1. Viral load 2M IU/mL. Sofosbuvir-ledipasvir 12-week course planned."),
        ("Gastroenterology","Irritable bowel syndrome. Rome IV criteria met. Low-FODMAP diet counseling. Rifaximin course prescribed."),
        ("Pulmonology",     "Severe COPD exacerbation. FEV1 35% predicted. Started on inhaled LABA/ICS. Pulmonary rehab referral."),
        ("Pulmonology",     "Community-acquired pneumonia right lower lobe. Started on azithromycin and amoxicillin-clavulanate."),
        ("Pulmonology",     "Asthma uncontrolled. Step-up to high-dose ICS. Montelukast added. Peak flow diary initiated."),
        ("Pulmonology",     "Obstructive sleep apnea. AHI 42 events/hour. CPAP initiated at 10 cmH2O. Follow-up polysomnography in 3 months."),
        ("Pulmonology",     "Pulmonary embolism bilateral. Started on rivaroxaban. V/Q scan confirms perfusion defects. Hematology consult."),
        ("Urology",         "Benign prostatic hyperplasia. IPSS score 18. Started tamsulosin. Residual urine 120 mL on ultrasound."),
        ("Urology",         "Renal cell carcinoma left kidney. Partial nephrectomy performed. Pathology clear cell carcinoma stage T1b."),
        ("Urology",         "Recurrent UTI. Urine culture E. coli. Nitrofurantoin prophylaxis. Cystoscopy unremarkable."),
        ("Urology",         "Kidney stone 7mm ureteral. Ureteroscopy with laser lithotripsy. Stone analysis: calcium oxalate monohydrate."),
        ("Urology",         "Bladder cancer surveillance cystoscopy. No recurrence. Continued BCG instillation maintenance therapy."),
        ("Dermatology",     "Psoriasis plaque type. PASI score 12. Topical clobetasol, vitamin D analog. Methotrexate initiated."),
        ("Dermatology",     "Atopic dermatitis severe. Dupilumab started. Moisturization regimen. Avoidance of known triggers."),
        ("Dermatology",     "Melanoma excision right arm. Sentinel lymph node biopsy negative. Stage IB. Dermatology follow-up 3 months."),
        ("Dermatology",     "Acne vulgaris moderate. Adapalene gel and doxycycline prescribed. Sun protection counseling provided."),
        ("Dermatology",     "Contact dermatitis. Patch testing positive to nickel. Topical hydrocortisone. Allergen avoidance advised."),
        ("Endocrinology",   "Type 2 diabetes HbA1c 9.2%. Metformin dose increased. Empagliflozin added. Diabetic eye exam ordered."),
        ("Endocrinology",   "Hypothyroidism. TSH 12.5 mIU/L. Levothyroxine 100 mcg started. Repeat TSH in 6 weeks."),
        ("Endocrinology",   "Cushing syndrome. 24-hour urinary cortisol elevated. Pituitary MRI shows 6mm adenoma. Endocrinology referral."),
        ("Endocrinology",   "Hyperthyroidism Graves disease. FT4 elevated, TSH suppressed. Methimazole started, radioactive iodine discussed."),
        ("Endocrinology",   "Obesity management. BMI 38. Lifestyle counseling, low-calorie diet. Phentermine-topiramate ER initiated."),
    ]
    np.random.seed(42)
    
    records = []
    for _ in range(10):
        for specialty, text in demo_records:
            words = text.split()
            np.random.shuffle(words)
            records.append({"medical_specialty": specialty,
                             "transcription": " ".join(words) + " " + text})
    df = pd.DataFrame(records)
    print(f"  Demo mode: {len(df)} synthetic records, {df['medical_specialty'].nunique()} specialties.")
else:
    df = pd.read_csv(DATASET_PATH)
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")
    # Standardise column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    if 'medical_specialty' not in df.columns:
        # Try to auto-detect
        specialty_col = [c for c in df.columns if 'specialty' in c or 'category' in c]
        text_col      = [c for c in df.columns if 'transcription' in c or 'text' in c or 'note' in c]
        if specialty_col and text_col:
            df = df.rename(columns={specialty_col[0]: 'medical_specialty',
                                     text_col[0]:     'transcription'})
        else:
            print("  ❌  Could not find specialty/text columns. Check your CSV.")
            sys.exit(1)


df = df[['medical_specialty', 'transcription']].dropna()
df['medical_specialty'] = df['medical_specialty'].str.strip()
df['transcription']      = df['transcription'].astype(str).str.strip()
df = df[df['transcription'].str.len() > 50]


top_specialties = df['medical_specialty'].value_counts().head(8).index.tolist()
df = df[df['medical_specialty'].isin(top_specialties)].reset_index(drop=True)

print(f"  After filtering: {len(df)} records, {df['medical_specialty'].nunique()} specialties")
print(f"  Specialties: {', '.join(sorted(df['medical_specialty'].unique()))}")
print()


print("STEP 2: Exploratory Data Analysis (EDA)...")
print("-" * 50)

os.makedirs('charts', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('NLP Dataset — Exploratory Data Analysis', fontsize=14, fontweight='bold')


counts = df['medical_specialty'].value_counts()
colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
bars = axes[0].barh(counts.index, counts.values, color=colors, edgecolor='black', linewidth=0.6)
axes[0].set_title('Class Distribution (Specialties)', fontweight='bold')
axes[0].set_xlabel('Number of Records')
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 str(val), va='center', fontsize=8)


df['text_length'] = df['transcription'].str.len()
axes[1].hist(df['text_length'], bins=40, color='steelblue', edgecolor='black', alpha=0.8)
axes[1].set_title('Clinical Note Length Distribution', fontweight='bold')
axes[1].set_xlabel('Character Count')
axes[1].set_ylabel('Frequency')
axes[1].axvline(df['text_length'].median(), color='red', linestyle='--',
                label=f"Median={int(df['text_length'].median())}")
axes[1].legend()


df['word_count'] = df['transcription'].str.split().str.len()
word_means = df.groupby('medical_specialty')['word_count'].mean().sort_values()
axes[2].barh(word_means.index, word_means.values,
             color=plt.cm.Pastel1(np.linspace(0,1,len(word_means))), edgecolor='black', linewidth=0.6)
axes[2].set_title('Average Word Count per Specialty', fontweight='bold')
axes[2].set_xlabel('Mean Word Count')

plt.tight_layout()
plt.savefig('charts/01_nlp_eda.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Chart saved: charts/01_nlp_eda.png")
print(f"  Text length — mean: {df['text_length'].mean():.0f}, median: {df['text_length'].median():.0f}")
print()


print("STEP 3: Advanced NLP Text Preprocessing...")
print("-" * 50)

STOPWORDS = set(stopwords.words('english'))
MEDICAL_STOPWORDS = {'patient', 'history', 'noted', 'also', 'without',
                      'the', 'and', 'of', 'was', 'is', 'with', 'were',
                      'he', 'she', 'his', 'her', 'has', 'had', 'have'}
ALL_STOPS = STOPWORDS | MEDICAL_STOPWORDS
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
      1. Lowercase
      2. Remove numbers & special characters
      3. Tokenize
      4. Remove stopwords (English + medical)
      5. Lemmatize
    """
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in ALL_STOPS and len(t) > 2]
    return ' '.join(tokens)

print("  Applying preprocessing pipeline (this may take a moment)...")
df['clean_text'] = df['transcription'].apply(preprocess_text)

# Vocabulary stats
all_words = ' '.join(df['clean_text']).split()
vocab = set(all_words)
print(f"  Vocabulary size after preprocessing: {len(vocab):,} unique tokens")
print(f"  Total tokens: {len(all_words):,}")

# Top medical terms
from collections import Counter
word_freq = Counter(all_words)
top_words = word_freq.most_common(20)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('NLP Preprocessing Analysis', fontsize=14, fontweight='bold')

words_list, freq_list = zip(*top_words)
colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(words_list)))
axes[0].barh(list(reversed(words_list)), list(reversed(freq_list)),
             color=list(reversed(colors_bar)), edgecolor='black', linewidth=0.5)
axes[0].set_title('Top 20 Medical Terms (after preprocessing)', fontweight='bold')
axes[0].set_xlabel('Frequency')

# Text length before vs after
before_lens = df['text_length'].values
after_lens  = df['clean_text'].str.len().values
axes[1].scatter(before_lens, after_lens, alpha=0.4, color='steelblue', s=10)
axes[1].set_title('Text Length Before vs After Preprocessing', fontweight='bold')
axes[1].set_xlabel('Original Length (chars)')
axes[1].set_ylabel('Cleaned Length (chars)')
z = np.polyfit(before_lens, after_lens, 1)
p = np.poly1d(z)
x_line = np.linspace(before_lens.min(), before_lens.max(), 100)
axes[1].plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
axes[1].legend()

plt.tight_layout()
plt.savefig('charts/02_nlp_preprocessing.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Chart saved: charts/02_nlp_preprocessing.png\n")


print("STEP 4: Encoding Labels & Splitting Dataset...")
print("-" * 50)

le = LabelEncoder()
y  = le.fit_transform(df['medical_specialty'])
X  = df['clean_text'].values
num_classes = len(le.classes_)

print(f"  Classes ({num_classes}): {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
print()

print("STEP 5: Building Advanced NLP Pipeline (TF-IDF + Ensemble)...")
print("-" * 50)

# TF-IDF with both unigrams and bigrams for richer features
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),      # unigrams + bigrams
    max_features=15000,       # top 15K features
    sublinear_tf=True,        # log-TF scaling
    min_df=2,                 # ignore very rare terms
    max_df=0.95,              # ignore near-universal terms
    analyzer='word',
    strip_accents='unicode',
)


lr_clf  = LogisticRegression(max_iter=1000, C=5.0, solver='lbfgs', random_state=42)

svm_clf = CalibratedClassifierCV(
    LinearSVC(max_iter=2000, C=1.0, random_state=42), cv=3)
rf_clf  = RandomForestClassifier(n_estimators=200, max_depth=20,
                                  min_samples_split=4, random_state=42, n_jobs=-1)

# Soft-voting ensemble
ensemble = VotingClassifier(
    estimators=[('lr', lr_clf), ('svm', svm_clf), ('rf', rf_clf)],
    voting='soft', weights=[3, 2, 1]   # LR is strongest on text
)


nlp_pipeline = Pipeline([
    ('tfidf',    tfidf),
    ('ensemble', ensemble),
])

print("  Pipeline architecture:")
print("    TF-IDF Vectorizer (unigrams+bigrams, 15K features)")
print("    └─ Soft-Voting Ensemble:")
print("         ├─ Logistic Regression (weight=3)")
print("         ├─ Linear SVM / Calibrated (weight=2)")
print("         └─ Random Forest 200 trees (weight=1)")
print()


print("STEP 6: 5-Fold Stratified Cross-Validation...")
print("-" * 50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(nlp_pipeline, X_train, y_train,
                             cv=cv, scoring='accuracy', n_jobs=-1)

print(f"  CV Accuracy per fold: {[round(s*100,2) for s in cv_scores]}")
print(f"  Mean CV Accuracy    : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# Plot CV scores
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar([f'Fold {i+1}' for i in range(5)],
              cv_scores * 100,
              color=plt.cm.Blues(np.linspace(0.4, 0.9, 5)),
              edgecolor='black', linewidth=0.7)
ax.axhline(cv_scores.mean()*100, color='red', linestyle='--', linewidth=2,
           label=f'Mean: {cv_scores.mean()*100:.2f}%')
ax.set_title('5-Fold Cross-Validation Accuracy', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(50, 105)
for bar, score in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5,
            f'{score*100:.1f}%', ha='center', fontweight='bold', fontsize=10)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('charts/03_cross_validation.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Chart saved: charts/03_cross_validation.png\n")


print("STEP 7: Training Final Model on Full Training Set...")
print("-" * 50)
print("  Training... (please wait, ensemble takes ~2 min)\n")

nlp_pipeline.fit(X_train, y_train)
print("  Training completed!\n")


print("STEP 8: Evaluating on Test Set...")
print("-" * 50)

y_pred      = nlp_pipeline.predict(X_test)
y_pred_prob = nlp_pipeline.predict_proba(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec  = recall_score(y_test, y_pred,    average='weighted')
f1   = f1_score(y_test, y_pred,        average='weighted')

print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1-Score  : {f1*100:.2f}%")
print()
print("  Per-class Report:")
print(classification_report(y_test, y_pred,
      target_names=le.classes_, digits=3))


fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('NLP Model Performance Evaluation', fontsize=14, fontweight='bold')

metrics_vals = [acc, prec, rec, f1]
metrics_lbl  = ['Accuracy', 'Precision\n(weighted)', 'Recall\n(weighted)', 'F1-Score\n(weighted)']
mcolors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
bars2 = axes[0].bar(metrics_lbl, [v*100 for v in metrics_vals],
                    color=mcolors, edgecolor='black', linewidth=0.8)
axes[0].set_title('Performance Metrics (%)', fontweight='bold')
axes[0].set_ylabel('Score (%)')
axes[0].set_ylim(0, 115)
for bar, val in zip(bars2, metrics_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
                 f'{val*100:.1f}%', ha='center', fontweight='bold', fontsize=10)


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[1], linewidths=0.5)
axes[1].set_title('Confusion Matrix', fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].tick_params(axis='x', rotation=35, labelsize=8)
axes[1].tick_params(axis='y', rotation=0,  labelsize=8)

plt.tight_layout()
plt.savefig('charts/04_evaluation.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Chart saved: charts/04_evaluation.png")


per_class_f1 = f1_score(y_test, y_pred, average=None)
fig, ax = plt.subplots(figsize=(12, 5))
bars3 = ax.bar(le.classes_, per_class_f1 * 100,
               color=plt.cm.Set3(np.linspace(0, 1, num_classes)),
               edgecolor='black', linewidth=0.7)
ax.set_title('Per-Class F1-Score', fontsize=13, fontweight='bold')
ax.set_ylabel('F1-Score (%)')
ax.set_ylim(0, 115)
ax.tick_params(axis='x', rotation=30)
for bar, val in zip(bars3, per_class_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
            f'{val*100:.1f}%', ha='center', fontweight='bold', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('charts/05_per_class_f1.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Chart saved: charts/05_per_class_f1.png")


print()
print("STEP 9: Extracting Top TF-IDF Features per Specialty...")
print("-" * 50)
tfidf_fitted  = nlp_pipeline.named_steps['tfidf']
feature_names = np.array(tfidf_fitted.get_feature_names_out())


lr_fitted = nlp_pipeline.named_steps['ensemble'].estimators_[0]
n_show    = min(10, num_classes)

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle('Top Discriminative Terms per Medical Specialty (LR Coefficients)',
             fontsize=13, fontweight='bold')

for i, (ax, cls_name) in enumerate(zip(axes.flatten(), le.classes_)):
    if hasattr(lr_fitted, 'coef_'):
        coef = lr_fitted.coef_[i]
        top_idx  = coef.argsort()[-10:][::-1]
        top_terms = feature_names[top_idx]
        top_coefs = coef[top_idx]
        cols = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_terms)))
        ax.barh(top_terms[::-1], top_coefs[::-1], color=cols[::-1],
                edgecolor='black', linewidth=0.4)
        ax.set_title(cls_name, fontweight='bold', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.axvline(0, color='black', linewidth=0.5)
    else:
        ax.text(0.5, 0.5, cls_name, ha='center', va='center', transform=ax.transAxes)
        ax.set_title(cls_name)


for ax in axes.flatten()[len(le.classes_):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig('charts/06_top_features.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Chart saved: charts/06_top_features.png\n")


print("STEP 10: Saving NLP Model & Artefacts...")
print("-" * 50)

os.makedirs('models', exist_ok=True)
nlp_model_data = {
    'pipeline':      nlp_pipeline,
    'label_encoder': le,
    'classes':       list(le.classes_),
    'accuracy':      acc,
    'precision':     prec,
    'recall':        rec,
    'f1_score':      f1,
    'cv_mean':       cv_scores.mean(),
    'cv_std':        cv_scores.std(),
    'num_features':  len(tfidf_fitted.get_feature_names_out()),
    'model_desc':    'TF-IDF (unigrams+bigrams) + Soft-Voting Ensemble (LR+SVM+RF)',
    'demo_mode':     DEMO_MODE,
}

with open('models/nlp_model.pkl', 'wb') as f:
    pickle.dump(nlp_model_data, f)

print(f"  Saved: models/nlp_model.pkl  ({os.path.getsize('models/nlp_model.pkl')//1024} KB)")


print()
print("=" * 65)
print("  NLP TRAINING COMPLETE — SUMMARY")
print("=" * 65)
print(f"  Dataset       : {'mtsamples.csv (Kaggle)' if not DEMO_MODE else 'Demo mode'}")
print(f"  Records       : {len(df)} clinical notes")
print(f"  Specialties   : {num_classes}")
print(f"  Features      : TF-IDF {len(tfidf_fitted.get_feature_names_out()):,} (uni+bigrams)")
print(f"  Model         : Soft-Voting Ensemble (LR + LinearSVM + RF-200)")
print(f"  CV Accuracy   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"  Test Accuracy : {acc*100:.2f}%")
print(f"  Test F1       : {f1*100:.2f}%")
print()
print("  Saved artefacts:")
print("  - models/nlp_model.pkl")
print("  - charts/01_nlp_eda.png")
print("  - charts/02_nlp_preprocessing.png")
print("  - charts/03_cross_validation.png")
print("  - charts/04_evaluation.png")
print("  - charts/05_per_class_f1.png")
print("  - charts/06_top_features.png")
print()
print("  Next step → run: python train_image_model.py")
print("  Then     → run: python app.py")
print("=" * 65)