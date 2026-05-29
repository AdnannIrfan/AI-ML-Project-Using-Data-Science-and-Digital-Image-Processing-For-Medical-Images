import os, sys, pickle, warnings
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("  MedicalAI Pro — Image Model Training")
print("  Chest X-Ray Pneumonia Detection")
print("=" * 65)
print()

IMG_SIZE   = 64
KAGGLE_DIR = "chest_xray"


def load_kaggle_images(base_dir, split='train'):
    images, labels = [], []
    for label_idx, cls in enumerate(['NORMAL', 'PNEUMONIA']):
        folder = os.path.join(base_dir, split, cls)
        if not os.path.exists(folder):
            return None, None
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            path = os.path.join(folder, fname)
            img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label_idx)
    return np.array(images), np.array(labels)

print("STEP 1: Loading Dataset...")
print("-" * 50)

X_train_imgs, y_train = load_kaggle_images(KAGGLE_DIR, 'train')
X_test_imgs,  y_test  = load_kaggle_images(KAGGLE_DIR, 'test')

if X_train_imgs is None:
    print("  chest_xray/ folder not found — using synthetic dataset as fallback.")
    print("  Download real data from:")
    print("  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia\n")
    REAL_DATA = False

    np.random.seed(42)
    NUM = 800

    def make_normal(n):
        imgs = []
        for _ in range(n):
            img = np.random.randint(130, 200, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            for i in range(0, IMG_SIZE, 8):
                cv2.line(img, (0, i), (IMG_SIZE, i+2), int(np.random.randint(160,210)), 1)
            img = cv2.GaussianBlur(img, (3,3), 0)
            imgs.append(img)
        return np.array(imgs)

    def make_pneumonia(n):
        imgs = []
        for _ in range(n):
            img = np.random.randint(60, 140, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            for _ in range(np.random.randint(4, 9)):
                x = np.random.randint(8, IMG_SIZE-16)
                y = np.random.randint(8, IMG_SIZE-16)
                r = np.random.randint(6, 16)
                cv2.circle(img, (x,y), r, int(np.random.randint(30,80)), -1)
            img = cv2.GaussianBlur(img, (5,5), 0)
            imgs.append(img)
        return np.array(imgs)

    all_imgs   = np.concatenate([make_normal(NUM//2), make_pneumonia(NUM//2)])
    all_labels = np.array([0]*(NUM//2) + [1]*(NUM//2))

    idx = np.random.permutation(len(all_imgs))
    all_imgs, all_labels = all_imgs[idx], all_labels[idx]

    X_train_imgs, X_test_imgs, y_train, y_test = train_test_split(
        all_imgs, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
else:
    REAL_DATA = True
    print(f"  Kaggle chest_xray/ found!")

print(f"  Train: {len(X_train_imgs)} images | Test: {len(X_test_imgs)} images")
print(f"  Normal train: {np.sum(y_train==0)} | Pneumonia train: {np.sum(y_train==1)}")
print()


print("STEP 2: Image EDA...")
print("-" * 50)

os.makedirs('charts', exist_ok=True)

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.suptitle('Sample Chest X-Ray Images — Normal vs Pneumonia', fontsize=13, fontweight='bold')

norm_idx  = np.where(y_train == 0)[0][:5]
pneu_idx  = np.where(y_train == 1)[0][:5]

for i, idx in enumerate(norm_idx):
    axes[0, i].imshow(X_train_imgs[idx], cmap='gray')
    axes[0, i].set_title('Normal', color='#2196F3', fontsize=9, fontweight='bold')
    axes[0, i].axis('off')

for i, idx in enumerate(pneu_idx):
    axes[1, i].imshow(X_train_imgs[idx], cmap='gray')
    axes[1, i].set_title('Pneumonia', color='#F44336', fontsize=9, fontweight='bold')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('charts/07_image_samples.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Chart saved: charts/07_image_samples.png")


print()
print("STEP 3: Advanced Feature Extraction (DIP + Statistical)...")
print("-" * 50)

def extract_features(img):
    """
    28-dimensional hand-crafted feature vector:
    - 5 global statistics
    - Laplacian variance (sharpness)
    - Edge density (Canny)
    - 8-bin histogram
    - GLCM texture: contrast, homogeneity, energy, correlation
    - 8 LBP-like texture features
    """
    blur     = cv2.GaussianBlur(img, (3,3), 0)
    enhanced = cv2.equalizeHist(blur)
    norm     = enhanced.astype(np.float32) / 255.0

    
    mean_v  = np.mean(norm)
    std_v   = np.std(norm)
    max_v   = np.max(norm)
    min_v   = np.min(norm)
    med_v   = np.median(norm)
    skew_v  = float(np.mean(((norm - mean_v)/( std_v+1e-8))**3))
    kurt_v  = float(np.mean(((norm - mean_v)/(std_v+1e-8))**4))

  
    lap_var = cv2.Laplacian((norm*255).astype(np.uint8), cv2.CV_64F).var()

   
    edges = cv2.Canny((norm*255).astype(np.uint8), 50, 150)
    edge_d = np.sum(edges>0) / edges.size

    
    hist = cv2.calcHist([(norm*255).astype(np.uint8)],[0],None,[8],[0,256])
    hist = (hist.flatten() / hist.sum()).tolist()

   
    def grad_stats(a, b):
        diff = (a.astype(float) - b.astype(float))**2
        return np.mean(diff), np.std(diff)
    p = (norm*255).astype(np.uint8)
    t1m, t1s = grad_stats(p[1:, :], p[:-1,:])   # vertical
    t2m, t2s = grad_stats(p[:,1:],  p[:,:-1])    # horizontal
    t3m, t3s = grad_stats(p[1:,1:], p[:-1,:-1])  # diagonal
    t4m, t4s = grad_stats(p[1:,:-1],p[:-1,1:])   # anti-diag

    features = np.array([
        mean_v, std_v, max_v, min_v, med_v, skew_v, kurt_v,
        lap_var, edge_d,
        *hist,
        t1m, t1s, t2m, t2s, t3m, t3s, t4m, t4s
    ])
    return features

print("  Extracting features from training images...")
X_train_feats = np.array([extract_features(img) for img in X_train_imgs])
X_test_feats  = np.array([extract_features(img) for img in X_test_imgs])

print(f"  Feature vector size: {X_train_feats.shape[1]} per image")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_feats)
X_test_sc  = scaler.transform(X_test_feats)


print()
print("STEP 4: Training Image Ensemble Model...")
print("-" * 50)

rf_img  = RandomForestClassifier(n_estimators=300, max_depth=None,
                                   min_samples_split=3, random_state=42, n_jobs=-1)
gb_img  = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                      max_depth=5, random_state=42)
svm_img = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)

ensemble_img = VotingClassifier(
    estimators=[('rf', rf_img), ('gb', gb_img), ('svm', svm_img)],
    voting='soft', weights=[2, 2, 1]
)

print("  Architecture: RF-300 + GBM-150 + SVM-RBF (soft voting)")


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_sc = cross_val_score(ensemble_img, X_train_sc, y_train,
                         cv=cv, scoring='accuracy', n_jobs=-1)
print(f"  CV Accuracy: {cv_sc.mean()*100:.2f}% ± {cv_sc.std()*100:.2f}%")

ensemble_img.fit(X_train_sc, y_train)
print("  Training complete!")


print()
print("STEP 5: Evaluation...")
print("-" * 50)

y_pred      = ensemble_img.predict(X_test_sc)
y_pred_prob = ensemble_img.predict_proba(X_test_sc)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_pred_prob)

print(f"  Accuracy : {acc*100:.2f}%")
print(f"  Precision: {prec*100:.2f}%")
print(f"  Recall   : {rec*100:.2f}%")
print(f"  F1-Score : {f1*100:.2f}%")
print(f"  AUC-ROC  : {auc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=['Normal','Pneumonia']))


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Image Model — Evaluation', fontsize=14, fontweight='bold')

metrics_v = [acc, prec, rec, f1, auc]
metrics_l = ['Accuracy','Precision','Recall','F1','AUC-ROC']
bars = axes[0].bar(metrics_l, [v*100 for v in metrics_v],
                   color=['#4CAF50','#2196F3','#FF9800','#9C27B0','#FF5722'],
                   edgecolor='black')
axes[0].set_title('Metrics (%)', fontweight='bold')
axes[0].set_ylim(0, 115)
for bar, val in zip(bars, metrics_v):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f'{val*100:.1f}%', ha='center', fontweight='bold', fontsize=9)

cm = confusion_matrix(y_test, y_pred)
im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_title('Confusion Matrix', fontweight='bold')
axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
axes[1].set_xticklabels(['Normal','Pneumonia'])
axes[1].set_yticklabels(['Normal','Pneumonia'])
thresh = cm.max()/2
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm[i,j], ha='center', va='center',
                     color='white' if cm[i,j]>thresh else 'black',
                     fontsize=16, fontweight='bold')
plt.colorbar(im, ax=axes[1])

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[2].plot(fpr, tpr, color='#2196F3', linewidth=2, label=f'AUC={auc:.3f}')
axes[2].plot([0,1],[0,1],'k--',linewidth=1, label='Random')
axes[2].fill_between(fpr, tpr, alpha=0.15, color='#2196F3')
axes[2].set_title('ROC Curve', fontweight='bold')
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/08_image_evaluation.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Chart saved: charts/08_image_evaluation.png\n")


print("STEP 6: Saving Image Model...")
os.makedirs('models', exist_ok=True)

img_model_data = {
    'model':        ensemble_img,
    'scaler':       scaler,
    'img_size':     IMG_SIZE,
    'feature_size': X_train_feats.shape[1],
    'classes':      ['Normal', 'Pneumonia'],
    'accuracy':     acc,
    'precision':    prec,
    'recall':       rec,
    'f1_score':     f1,
    'auc_roc':      auc,
    'cv_mean':      cv_sc.mean(),
    'real_data':    REAL_DATA,
}

with open('models/image_model.pkl', 'wb') as f:
    pickle.dump(img_model_data, f)

print(f"  Saved: models/image_model.pkl")
print()
print("=" * 65)
print("  IMAGE MODEL TRAINING COMPLETE")
print("=" * 65)
print(f"  Accuracy : {acc*100:.2f}%  |  AUC-ROC: {auc:.4f}")
print()
print("  Next → run: python app.py")
print("=" * 65)
