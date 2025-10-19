import os
import json
import requests
from datetime import datetime

try:
    import firebase_admin
    from firebase_admin import credentials
    from firebase_admin import db as rtdb
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

PROJECT_ID = 'diabetes-prediction-22082'
DATABASE_URL = 'https://diabetes-prediction-22082-default-rtdb.firebaseio.com'
WEB_API_KEY = 'AIzaSyDQ70lgR3Vk5ykOWyUxKBQ3J-6p4dMKlxw'
firebase_initialized = False
db_ref = None
use_rest_api = False

# ==================== FIREBASE REST API ====================

class FirebaseRestDB:
    """Firebase Realtime Database using REST API"""
    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')
        print(f'✅ Connected to Firebase Realtime Database (REST API)')
        print(f'✅ Database: {self.base_url}')
    
    def child(self, path):
        return FirebaseRestRef(self.base_url, path)
    
    def get(self):
        try:
            response = requests.get(f'{self.base_url}/.json')
            if response.status_code == 200:
                return response.json() or {}
            return {}
        except Exception as e:
            print(f'Firebase GET error: {e}')
            return {}
    
    def set(self, value):
        try:
            response = requests.put(f'{self.base_url}/.json', json=value)
            return response.status_code == 200
        except Exception as e:
            print(f'Firebase SET error: {e}')
            return False

class FirebaseRestRef:
    """Firebase Realtime Database reference using REST API"""
    def __init__(self, base_url, path):
        self.base_url = base_url
        self.path = path
        self.full_path = f"{base_url}/{path}.json" if path else f"{base_url}/.json"
    
    def child(self, sub_path):
        new_path = f"{self.path}/{sub_path}" if self.path else sub_path
        return FirebaseRestRef(self.base_url, new_path)
    
    def get(self):
        try:
            response = requests.get(self.full_path)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f'Firebase GET error at {self.path}: {e}')
            return None
    
    def set(self, value):
        try:
            response = requests.put(self.full_path, json=value)
            return response.status_code == 200
        except Exception as e:
            print(f'Firebase SET error at {self.path}: {e}')
            return False
    
    def update(self, value):
        try:
            response = requests.patch(self.full_path, json=value)
            return response.status_code == 200
        except Exception as e:
            print(f'Firebase UPDATE error at {self.path}: {e}')
            return False
    
    def push(self, value=None):
        try:
            response = requests.post(self.full_path, json=value)
            if response.status_code == 200:
                data = response.json()
                push_id = data.get('name', '')
                return FirebaseRestRef(self.base_url, f"{self.path}/{push_id}" if self.path else push_id)
            return None
        except Exception as e:
            print(f'Firebase PUSH error at {self.path}: {e}')
            return None
    
    def delete(self):
        """Delete data at this reference"""
        try:
            response = requests.delete(self.full_path)
            return response.status_code == 200
        except Exception as e:
            print(f'Firebase DELETE error at {self.path}: {e}')
            return False

# ==================== LOCAL STORAGE (FALLBACK) ====================

class LocalDB:
    def __init__(self):
        self.data = {}
        print('⚠️ Using Local Storage Mode (Temporary)')
    def child(self, path):
        return LocalDBRef(self.data, path)
    def get(self):
        return self.data
    def set(self, value):
        self.data = value

class LocalDBRef:
    def __init__(self, data, path):
        self.data = data
        self.path = path
    def child(self, sub_path):
        new_path = f'{self.path}/{sub_path}' if self.path else sub_path
        return LocalDBRef(self.data, new_path)
    def get(self):
        parts = self.path.split('/') if self.path else []
        current = self.data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
    def set(self, value):
        parts = self.path.split('/') if self.path else []
        if not parts:
            self.data.update(value if isinstance(value, dict) else {})
            return
        current = self.data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    def update(self, value):
        parts = self.path.split('/') if self.path else []
        if not parts:
            if isinstance(value, dict):
                self.data.update(value)
            return
        current = self.data
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        leaf = parts[-1]
        if leaf not in current or not isinstance(current[leaf], dict):
            current[leaf] = {}
        if isinstance(value, dict):
            current[leaf].update(value)
        else:
            current[leaf] = value

local_db = LocalDB()
db_ref = local_db

def initialize_firebase():
    """Initialize Firebase - try REST API first, then Admin SDK, then local storage"""
    global firebase_initialized, db_ref, use_rest_api
    
    if firebase_initialized:
        return True
    
    # Try REST API (works with Web API Key - no service account needed!)
    try:
        test_ref = FirebaseRestDB(DATABASE_URL)
        # Test connection
        test_data = test_ref.get()
        db_ref = test_ref
        firebase_initialized = True
        use_rest_api = True
        print('🔥 Using Firebase REST API (Real Database Connected!)')
        return True
    except Exception as e:
        print(f'⚠️ REST API connection failed: {e}')
    
    # Try Admin SDK with service account
    if FIREBASE_AVAILABLE:
        try:
            if os.path.exists('firebase-service-account.json'):
                cred = credentials.Certificate('firebase-service-account.json')
                firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})
                db_ref = rtdb.reference()
                firebase_initialized = True
                use_rest_api = False
                print('✅ Firebase Admin SDK connected!')
                return True
        except Exception as e:
            print(f'⚠️ Admin SDK error: {e}')
    
    # Fallback to local storage
    print('⚠️ Using Local Storage Mode (data will not persist)')
    return False

def save_patient_data(patient_info, prediction_data, user_id=None):
    global db_ref
    try:
        timestamp = datetime.now()
        doc_id = f'pred_{timestamp.strftime("%Y%m%d%H%M%S%f")}'
        
        # Extract features array
        features = prediction_data.get('features', [])
        
        # Ensure we have 8 features
        while len(features) < 8:
            features.append(0)
        
        data = {
            'patient_info': patient_info,
            'patient_name': patient_info.get('name', 'Unknown'),
            'age': patient_info.get('age', 0),
            'sex': patient_info.get('sex', 'Unknown'),
            'contact': patient_info.get('contact', 'N/A'),
            'address': patient_info.get('address', 'N/A'),
            'prediction': prediction_data.get('prediction'),
            'result': prediction_data.get('risk_level', '').replace('high', 'High Risk').replace('low', 'Low Risk'),
            'risk_level': prediction_data.get('risk_level'),
            'confidence': prediction_data.get('confidence'),
            'features': features,
            # Individual parameters for easy access
            'Pregnancies': features[0] if len(features) > 0 else 0,
            'Glucose': features[1] if len(features) > 1 else 0,
            'BloodPressure': features[2] if len(features) > 2 else 0,
            'SkinThickness': features[3] if len(features) > 3 else 0,
            'Insulin': features[4] if len(features) > 4 else 0,
            'BMI': features[5] if len(features) > 5 else 0,
            'DiabetesPedigreeFunction': features[6] if len(features) > 6 else 0,
            'Age': features[7] if len(features) > 7 else patient_info.get('age', 0),
            # Lifestyle parameters (default to 0/No)
            'smoking': 0,
            'physical_activity': 0,
            'alcohol_intake': 0,
            'family_history': 0,
            'sleep_hours': 7,
            'user_id': user_id or 'anonymous',
            'timestamp': timestamp.isoformat(),
            'date': timestamp.strftime('%Y-%m-%d'),
            'time': timestamp.strftime('%H:%M:%S'),
            'created_at': timestamp.timestamp(),
            'report_id': doc_id
        }
        db_ref.child('predictions').child(doc_id).set(data)
        if user_id and user_id != 'anonymous':
            user_data = {
                'prediction_id': doc_id,
                'report_id': doc_id,
                'timestamp': data['timestamp'],
                'risk_level': data['risk_level'],
                'patient_name': data['patient_name'],
                'result': data['result'],
                'confidence': data['confidence'],
                # Include all parameters for user's predictions too
                'Pregnancies': data['Pregnancies'],
                'Glucose': data['Glucose'],
                'BloodPressure': data['BloodPressure'],
                'SkinThickness': data['SkinThickness'],
                'Insulin': data['Insulin'],
                'BMI': data['BMI'],
                'DiabetesPedigreeFunction': data['DiabetesPedigreeFunction'],
                'Age': data['Age'],
                'smoking': data['smoking'],
                'physical_activity': data['physical_activity'],
                'alcohol_intake': data['alcohol_intake'],
                'family_history': data['family_history'],
                'sleep_hours': data['sleep_hours']
            }
            db_ref.child('users').child(user_id).child('predictions').child(doc_id).set(user_data)
        return doc_id
    except Exception as e:
        print(f"Error saving patient data: {e}")
        return None

def get_patient_history(patient_name=None, user_id=None, limit=50):
    global db_ref
    try:
        predictions = db_ref.child('predictions').get()
        if not predictions:
            return []
        history = []
        for pred_id, pred_data in predictions.items():
            if not isinstance(pred_data, dict):
                continue
            if user_id and pred_data.get('user_id') != user_id:
                continue
            if patient_name and pred_data.get('patient_name', '').lower() != patient_name.lower():
                continue
            pred_data['id'] = pred_id
            history.append(pred_data)
        history.sort(key=lambda x: x.get('created_at', 0), reverse=True)
        return history[:limit]
    except Exception as e:
        return []

def get_statistics(user_id=None):
    global db_ref
    try:
        predictions = db_ref.child('predictions').get()
        if not predictions:
            return {'total_predictions': 0, 'high_risk_count': 0, 'low_risk_count': 0, 'average_confidence': 0, 'average_risk_percentage': 0}
        total = 0
        high_risk = 0
        low_risk = 0
        total_confidence = 0
        for pred_id, pred_data in predictions.items():
            if not isinstance(pred_data, dict):
                continue
            if user_id and pred_data.get('user_id') != user_id:
                continue
            total += 1
            if pred_data.get('risk_level') == 'high':
                high_risk += 1
            else:
                low_risk += 1
            total_confidence += pred_data.get('confidence', 0)
        avg_confidence = round(total_confidence / total, 2) if total > 0 else 0
        avg_risk = round((high_risk / total * 100), 2) if total > 0 else 0
        stats = {'total_predictions': total, 'high_risk_count': high_risk, 'low_risk_count': low_risk, 'average_confidence': avg_confidence, 'average_risk_percentage': avg_risk}
        if user_id and user_id != 'anonymous':
            db_ref.child('users').child(user_id).child('statistics').set(stats)
        return stats
    except Exception as e:
        return {'total_predictions': 0, 'high_risk_count': 0, 'low_risk_count': 0, 'average_confidence': 0, 'average_risk_percentage': 0}

def get_user_predictions(user_id=None, limit=50):
    """Get predictions for a specific user"""
    return get_patient_history(user_id=user_id, limit=limit)

def get_user_statistics(user_id=None):
    """Get statistics for a specific user"""
    return get_statistics(user_id=user_id)


def get_prediction_by_id(prediction_id):
    """Fetch a single prediction document by its ID"""
    global db_ref
    if not prediction_id:
        return None
    try:
        initialize_firebase()
        return db_ref.child('predictions').child(prediction_id).get()
    except Exception as e:
        print(f"Error fetching prediction {prediction_id}: {e}")
        return None


def get_predictions_by_ids(prediction_ids):
    """Fetch multiple predictions at once"""
    results = {}
    if not prediction_ids:
        return results
    for pred_id in prediction_ids:
        document = get_prediction_by_id(pred_id)
        if document:
            results[pred_id] = document
    return results


def update_prediction_record(prediction_id, updates, user_id=None):
    """Update a prediction record in the global and user-specific nodes"""
    global db_ref
    if not prediction_id or not isinstance(updates, dict) or not updates:
        return False
    try:
        initialize_firebase()
        prediction_ref = db_ref.child('predictions').child(prediction_id)
        prediction_ref.update(updates)
        if user_id and user_id != 'anonymous':
            user_ref = db_ref.child('users').child(user_id).child('predictions').child(prediction_id)
            user_ref.update(updates)
        return True
    except Exception as e:
        print(f"Error updating prediction {prediction_id}: {e}")
        return False


def append_prediction_comparison(prediction_id, comparison_entry, user_id=None):
    """Append a comparison analysis entry under a prediction document"""
    global db_ref
    if not prediction_id or not isinstance(comparison_entry, dict):
        return False
    try:
        initialize_firebase()
        prediction_ref = db_ref.child('predictions').child(prediction_id)
        existing = prediction_ref.get() or {}
        comparisons = existing.get('comparisons', {}) if isinstance(existing, dict) else {}
        analysis_id = comparison_entry.get('analysis_id')
        if not analysis_id:
            return False
        comparisons[analysis_id] = comparison_entry
        prediction_ref.update({'comparisons': comparisons})
        if user_id and user_id != 'anonymous':
            user_ref = db_ref.child('users').child(user_id).child('predictions').child(prediction_id)
            user_doc = user_ref.get() or {}
            user_comparisons = user_doc.get('comparisons', {}) if isinstance(user_doc, dict) else {}
            user_comparisons[analysis_id] = comparison_entry
            user_ref.update({'comparisons': user_comparisons})
        return True
    except Exception as e:
        print(f"Error storing comparison for prediction {prediction_id}: {e}")
        return False

# ==================== FIRESTORE-LIKE CLASSES ====================

class LocalCollection:
    """Firestore-like collection for local storage with Firebase sync"""
    def __init__(self, data, collection_name):
        self.data = data
        self.collection_name = collection_name
        if collection_name not in self.data:
            self.data[collection_name] = {}
    
    def _sync_to_firebase(self):
        """Sync collection data back to Firebase"""
        if use_rest_api and firebase_initialized:
            try:
                db_ref.child(self.collection_name).set(self.data[self.collection_name])
            except Exception as e:
                print(f'Sync error: {e}')
    
    def document(self, doc_id):
        return LocalDocument(self.data[self.collection_name], doc_id, None, self.collection_name)
    
    def add(self, data):
        """Add a new document with auto-generated ID"""
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.data[self.collection_name][doc_id] = data
        self._sync_to_firebase()
        return (datetime.now(), LocalDocument(self.data[self.collection_name], doc_id, data, self.collection_name))
    
    def where(self, field, op, value):
        return LocalQuery(self.data[self.collection_name], field, op, value, self.collection_name)
    
    def stream(self):
        """Return all documents as a stream"""
        return [LocalDocument(self.data[self.collection_name], doc_id, doc_data, self.collection_name) 
                for doc_id, doc_data in self.data[self.collection_name].items()]
    
    def get(self):
        """Get all documents"""
        return [LocalDocument(self.data[self.collection_name], doc_id, doc_data, self.collection_name) 
                for doc_id, doc_data in self.data[self.collection_name].items()]

class LocalDocument:
    """Firestore-like document for local storage with Firebase sync"""
    def __init__(self, collection_data, doc_id, doc_data=None, collection_name=None):
        self.collection_data = collection_data
        self.collection_name = collection_name
        self.doc_id = doc_id
        self.id = doc_id
        self._data = doc_data
        if doc_id not in self.collection_data and doc_data is None:
            self.collection_data[doc_id] = {}
    
    def _sync_to_firebase(self):
        """Sync document to Firebase"""
        if use_rest_api and firebase_initialized and self.collection_name:
            try:
                db_ref.child(self.collection_name).child(self.doc_id).set(self.collection_data[self.doc_id])
            except Exception as e:
                print(f'Sync error: {e}')
    
    def get(self):
        """Get document data"""
        return self
    
    def to_dict(self):
        """Return document data as dict"""
        if self._data:
            return self._data
        return self.collection_data.get(self.doc_id, {})
    
    def set(self, data, merge=False):
        """Set document data"""
        if merge and self.doc_id in self.collection_data:
            self.collection_data[self.doc_id].update(data)
        else:
            self.collection_data[self.doc_id] = data
        self._sync_to_firebase()
    
    def update(self, data):
        """Update document data"""
        if self.doc_id in self.collection_data:
            self.collection_data[self.doc_id].update(data)
        else:
            self.collection_data[self.doc_id] = data
        self._sync_to_firebase()
    
    def delete(self):
        """Delete document"""
        if self.doc_id in self.collection_data:
            del self.collection_data[self.doc_id]
    
    @property
    def exists(self):
        """Check if document exists"""
        return self.doc_id in self.collection_data

class LocalQuery:
    """Firestore-like query for local storage"""
    def __init__(self, collection_data, field, op, value, collection_name=None):
        self.collection_data = collection_data
        self.collection_name = collection_name
        self.field = field
        self.op = op
        self.value = value
        self._limit = None
    
    def limit(self, count):
        """Limit the number of results"""
        self._limit = count
        return self
    
    def stream(self):
        """Return filtered documents"""
        results = []
        for doc_id, doc_data in self.collection_data.items():
            if not isinstance(doc_data, dict):
                continue
            
            field_value = doc_data.get(self.field)
            
            if self.op == '==':
                if field_value == self.value:
                    results.append(LocalDocument(self.collection_data, doc_id, doc_data, self.collection_name))
            elif self.op == '!=':
                if field_value != self.value:
                    results.append(LocalDocument(self.collection_data, doc_id, doc_data, self.collection_name))
            elif self.op == '>':
                if field_value and field_value > self.value:
                    results.append(LocalDocument(self.collection_data, doc_id, doc_data, self.collection_name))
            elif self.op == '>=':
                if field_value and field_value >= self.value:
                    results.append(LocalDocument(self.collection_data, doc_id, doc_data, self.collection_name))
            elif self.op == '<':
                if field_value and field_value < self.value:
                    results.append(LocalDocument(self.collection_data, doc_id, doc_data, self.collection_name))
            elif self.op == '<=':
                if field_value and field_value <= self.value:
                    results.append(LocalDocument(self.collection_data, doc_id, doc_data, self.collection_name))
        
        # Apply limit if set
        if self._limit is not None:
            results = results[:self._limit]
        
        return results


class DBWrapper:
    """Wrapper supporting both Realtime DB and Firestore syntax"""
    def __init__(self):
        # Use Firebase data if connected, otherwise local storage
        if use_rest_api and firebase_initialized:
            # For REST API, we need to fetch data for Firestore-like operations
            try:
                self.data = db_ref.get() or {}
            except:
                self.data = {}
        else:
            self.data = local_db.data if hasattr(local_db, 'data') else {}
    
    def child(self, path):
        """Realtime Database style"""
        return db_ref.child(path)
    
    def collection(self, collection_name):
        """Firestore style - uses local collection wrapper for Firebase data"""
        # Refresh data from Firebase if using REST API
        if use_rest_api and firebase_initialized:
            try:
                self.data = db_ref.get() or {}
            except:
                pass
        return LocalCollection(self.data, collection_name)
    
    def get(self):
        return db_ref.get()
    
    def set(self, value):
        return db_ref.set(value)

# Initialize Firebase first
initialize_firebase()

# Create DB wrapper after initialization
db = DBWrapper()

