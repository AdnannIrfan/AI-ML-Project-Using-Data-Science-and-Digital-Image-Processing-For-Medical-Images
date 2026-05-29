from flask import Flask, render_template, request, redirect, url_for, jsonify
import os, pickle, base64, io, re
import numpy as np
import cv2
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')


for pkg in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

app = Flask(__name__)
app.config['UPLOAD_FOLDER']      = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs('uploads', exist_ok=True)

ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}


IMAGE_MODEL = None
NLP_MODEL   = None

if os.path.exists('models/image_model.pkl'):
    with open('models/image_model.pkl', 'rb') as f:
        IMAGE_MODEL = pickle.load(f)
    print("✅ Image model loaded")
else:
    print("⚠️  models/image_model.pkl not found — run train_image_model.py first")

if os.path.exists('models/nlp_model.pkl'):
    with open('models/nlp_model.pkl', 'rb') as f:
        NLP_MODEL = pickle.load(f)
    print("✅ NLP model loaded")
else:
    print("⚠️  models/nlp_model.pkl not found — run train_nlp_model.py first")


STOPWORDS   = set(stopwords.words('english'))
MEDICAL_STOPS = {'patient','history','noted','also','without','the','and',
                 'of','was','is','with','were','he','she','his','her','has','had','have'}
ALL_STOPS   = STOPWORDS | MEDICAL_STOPS
lemmatizer  = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in ALL_STOPS and len(t) > 2]
    return ' '.join(tokens)


IMG_SIZE = 64

def extract_image_features(img_gray):
    blur     = cv2.GaussianBlur(img_gray, (3, 3), 0)
    enhanced = cv2.equalizeHist(blur)
    norm     = enhanced.astype(np.float32) / 255.0
    mean_v = np.mean(norm); std_v = np.std(norm)
    max_v  = np.max(norm);  min_v = np.min(norm); med_v = np.median(norm)
    skew_v = float(np.mean(((norm - mean_v)/(std_v+1e-8))**3))
    kurt_v = float(np.mean(((norm - mean_v)/(std_v+1e-8))**4))
    lap_var = cv2.Laplacian((norm*255).astype(np.uint8), cv2.CV_64F).var()
    edges   = cv2.Canny((norm*255).astype(np.uint8), 50, 150)
    edge_d  = np.sum(edges > 0) / edges.size
    hist    = cv2.calcHist([(norm*255).astype(np.uint8)], [0], None, [8], [0,256])
    hist    = (hist.flatten() / hist.sum()).tolist()
    p = (norm*255).astype(np.uint8)
    def gstat(a, b):
        d = (a.astype(float)-b.astype(float))**2
        return np.mean(d), np.std(d)
    t1m,t1s = gstat(p[1:,:],   p[:-1,:])
    t2m,t2s = gstat(p[:,1:],   p[:,:-1])
    t3m,t3s = gstat(p[1:,1:],  p[:-1,:-1])
    t4m,t4s = gstat(p[1:,:-1], p[:-1,1:])
    return np.array([mean_v,std_v,max_v,min_v,med_v,skew_v,kurt_v,
                     lap_var,edge_d,*hist,t1m,t1s,t2m,t2s,t3m,t3s,t4m,t4s])

def img_to_b64(arr, color=True):
    if arr is None: return ""
    pil = Image.fromarray(arr if color else arr)
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def allowed_file(fname):
    return '.' in fname and fname.rsplit('.',1)[1].lower() in ALLOWED_EXT


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('upload.html', error='No file selected.')
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return render_template('upload.html', error='Invalid file type.')

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

      
        img_bgr   = cv2.imread(filepath)
        img_rs    = cv2.resize(img_bgr, (224,224))
        img_gray  = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
        img_blur  = cv2.GaussianBlur(img_gray, (5,5), 0)
        img_enh   = cv2.equalizeHist(img_blur)
        img_edges = cv2.Canny(img_enh, 50, 150)

        orig_b64  = img_to_b64(cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB))
        gray_b64  = img_to_b64(img_gray,  color=False)
        enh_b64   = img_to_b64(img_enh,   color=False)
        edge_b64  = img_to_b64(img_edges, color=False)

        
        mean_v = float(np.mean(img_gray))
        features = {
            'mean':       round(mean_v, 2),
            'std':        round(float(np.std(img_gray)), 2),
            'contrast':   round(float(np.max(img_gray)-np.min(img_gray)), 2),
            'entropy':    round(float(-np.sum(
                              (h:=cv2.calcHist([img_gray],[0],None,[256],[0,256])/img_gray.size+1e-10)
                              * np.log2(h+1e-10))), 2),
            'brightness': 'High' if mean_v > 128 else 'Low'
        }

        
        if IMAGE_MODEL:
            img_sm  = cv2.resize(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
            feats   = extract_image_features(img_sm).reshape(1,-1)
            feats_s = IMAGE_MODEL['scaler'].transform(feats)
            pred    = IMAGE_MODEL['model'].predict(feats_s)[0]
            probs   = IMAGE_MODEL['model'].predict_proba(feats_s)[0]
            cls     = IMAGE_MODEL['classes'][pred]
            result  = {
                'disease_label':   'Chest X-Ray Pneumonia Detection',
                'predicted_class': cls,
                'confidence':      round(float(probs[pred])*100, 2),
                'probabilities':   {'Normal': round(float(probs[0])*100,2),
                                    'Pneumonia': round(float(probs[1])*100,2)},
                'metrics': {
                    'accuracy':  round(IMAGE_MODEL['accuracy']*100,2),
                    'precision': round(IMAGE_MODEL['precision']*100,2),
                    'recall':    round(IMAGE_MODEL['recall']*100,2),
                    'f1_score':  round(IMAGE_MODEL['f1_score']*100,2),
                },
                'model_type': 'RF-300 + GBM-150 + SVM-RBF Ensemble'
            }
        else:
            result = {
                'disease_label':   'Chest X-Ray (Demo Mode)',
                'predicted_class': 'Model not trained yet',
                'confidence':      0,
                'probabilities':   {'Normal': 50, 'Pneumonia': 50},
                'metrics':         {'accuracy':0,'precision':0,'recall':0,'f1_score':0},
                'model_type':      'Run train_image_model.py first'
            }

        return render_template('result.html',
                               orig_img=orig_b64, gray_img=gray_b64,
                               enhanced_img=enh_b64, edges_img=edge_b64,
                               features=features, result=result,
                               filename=file.filename)
    return render_template('upload.html')


@app.route('/nlp', methods=['GET','POST'])
def nlp_page():
    prediction = None
    if request.method == 'POST':
        text = request.form.get('clinical_text','').strip()
        if not text:
            return render_template('nlp.html', error='Please enter clinical text.')

        if NLP_MODEL:
            clean   = preprocess_text(text)
            pipeline = NLP_MODEL['pipeline']
            le       = NLP_MODEL['label_encoder']
            probs    = pipeline.predict_proba([clean])[0]
            pred_idx = np.argmax(probs)
            classes  = NLP_MODEL['classes']

            top3_idx  = np.argsort(probs)[::-1][:3]
            top3      = [(classes[i], round(float(probs[i])*100,2)) for i in top3_idx]

            prediction = {
                'specialty':   classes[pred_idx],
                'confidence':  round(float(probs[pred_idx])*100, 2),
                'top3':        top3,
                'all_probs':   [(classes[i], round(float(probs[i])*100,2))
                                for i in np.argsort(probs)[::-1]],
                'metrics': {
                    'accuracy':  round(NLP_MODEL['accuracy']*100,2),
                    'f1_score':  round(NLP_MODEL['f1_score']*100,2),
                    'cv_mean':   round(NLP_MODEL['cv_mean']*100,2),
                },
                'model_type': NLP_MODEL['model_desc'],
                'word_count': len(text.split()),
                'clean_tokens': len(clean.split()),
            }
        else:
            prediction = {
                'specialty':  'Model Not Loaded',
                'confidence': 0,
                'top3': [],
                'all_probs': [],
                'metrics': {'accuracy':0,'f1_score':0,'cv_mean':0},
                'model_type': 'Run train_nlp_model.py first',
                'word_count': len(text.split()),
                'clean_tokens': 0,
            }

    return render_template('nlp.html', prediction=prediction)


@app.route('/about')
def about():
    img_info = IMAGE_MODEL or {}
    nlp_info = NLP_MODEL   or {}
    return render_template('about.html', img_model=img_info, nlp_model=nlp_info)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
