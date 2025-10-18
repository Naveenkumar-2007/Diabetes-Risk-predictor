"""
User Authentication and Authorization Module
Handles user registration, login, and role-based access control
"""
import os
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
from flask import session, redirect, url_for, request
from firebase_config import db

# Admin credentials (hardcoded for security)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_HASH = "e54fc6b51915e222ba6196747a19ebb8dfa651fd2b46a385a0ded647fbfefda0"  # Change this password!

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_session_token():
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)

def create_user(username, email, password, full_name, contact="", address=""):
    """
    Create a new user account
    
    Args:
        username: Unique username
        email: User email
        password: Plain text password (will be hashed)
        full_name: User's full name
        contact: Phone number (optional)
        address: Address (optional)
    
    Returns:
        Tuple (success: bool, message: str, user_id: str or None)
    """
    if not db:
        return False, "Database not initialized", None
    
    try:
        # Check if username already exists
        users_ref = db.collection('users')
        existing = users_ref.where('username', '==', username).limit(1).stream()
        
        if len(list(existing)) > 0:
            return False, "Username already exists", None
        
        # Check if email already exists
        existing_email = users_ref.where('email', '==', email).limit(1).stream()
        if len(list(existing_email)) > 0:
            return False, "Email already registered", None
        
        # Create user document
        user_data = {
            'username': username,
            'email': email,
            'password_hash': hash_password(password),
            'full_name': full_name,
            'contact': contact,
            'address': address,
            'role': 'user',  # Default role
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'is_active': True
        }
        
        doc_ref = users_ref.add(user_data)
        user_id = doc_ref[1].id
        
        print(f"✅ User created: {username} (ID: {user_id})")
        return True, "Account created successfully", user_id
        
    except Exception as e:
        print(f"❌ Error creating user: {e}")
        return False, f"Error: {str(e)}", None

def authenticate_user(username, password):
    """
    Authenticate user credentials
    
    Args:
        username: Username
        password: Plain text password
    
    Returns:
        Tuple (success: bool, message: str, user_data: dict or None)
    """
    # Check for admin login
    if username == ADMIN_USERNAME and hash_password(password) == ADMIN_PASSWORD_HASH:
        return True, "Admin login successful", {
            'username': ADMIN_USERNAME,
            'role': 'admin',
            'full_name': 'Administrator',
            'email': 'admin@diabetes-predictor.com',
            'user_id': 'admin'
        }
    
    if not db:
        return False, "Database not initialized", None
    
    try:
        # Find user by username
        users_ref = db.collection('users')
        users = users_ref.where('username', '==', username).limit(1).stream()
        
        user_doc = None
        for user in users:
            user_doc = user
            break
        
        if not user_doc:
            return False, "Invalid username or password", None
        
        user_data = user_doc.to_dict()
        user_id = user_doc.id
        
        # Check if account is active
        if not user_data.get('is_active', True):
            return False, "Account is disabled", None
        
        # Verify password
        password_hash = hash_password(password)
        if password_hash != user_data.get('password_hash'):
            return False, "Invalid username or password", None
        
        # Update last login
        try:
            user_doc.reference.update({
                'last_login': datetime.now().isoformat()
            })
        except:
            pass  # Non-critical
        
        # Return user data (without password hash)
        user_info = {
            'user_id': user_id,
            'username': user_data['username'],
            'email': user_data['email'],
            'full_name': user_data['full_name'],
            'contact': user_data.get('contact', ''),
            'address': user_data.get('address', ''),
            'role': user_data.get('role', 'user')
        }
        
        print(f"✅ User authenticated: {username}")
        return True, "Login successful", user_info
        
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return False, f"Error: {str(e)}", None

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator to require admin role for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        if session.get('role') != 'admin':
            return redirect(url_for('user_dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def change_admin_password(old_password, new_password):
    """Change admin password - saves to a file"""
    global ADMIN_PASSWORD_HASH
    
    # Verify old password
    if hash_password(old_password) != ADMIN_PASSWORD_HASH:
        return False, "Current password is incorrect"
    
    # Validate new password
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters"
    
    # Update the password hash
    ADMIN_PASSWORD_HASH = hash_password(new_password)
    
    # Save to file
    try:
        auth_file_path = os.path.join(os.path.dirname(__file__), 'auth.py')
        with open(auth_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the password hash line
        old_line = f'ADMIN_PASSWORD_HASH = "e54fc6b51915e222ba6196747a19ebb8dfa651fd2b46a385a0ded647fbfefda0"'
        new_line = f'ADMIN_PASSWORD_HASH = "{ADMIN_PASSWORD_HASH}"'
        
        # If old_line not found, try to find any ADMIN_PASSWORD_HASH line
        if old_line not in content:
            import re
            content = re.sub(
                r'ADMIN_PASSWORD_HASH = .*',
                new_line,
                content
            )
        else:
            content = content.replace(old_line, new_line)
        
        with open(auth_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True, "Admin password changed successfully"
    except Exception as e:
        return False, f"Error saving password: {str(e)}"

def change_user_password(user_id, old_password, new_password):
    """Change user password in Firebase"""
    try:
        # Get user data
        user_ref = db.child('users').child(user_id)
        user_data = user_ref.get()
        
        if not user_data:
            return False, "User not found"
        
        # Verify old password
        if hash_password(old_password) != user_data.get('password'):
            return False, "Current password is incorrect"
        
        # Validate new password
        if len(new_password) < 6:
            return False, "New password must be at least 6 characters"
        
        # Update password in Firebase
        user_ref.update({
            'password': hash_password(new_password)
        })
        
        return True, "Password changed successfully"
    except Exception as e:
        return False, f"Error changing password: {str(e)}"

def get_user_predictions(user_id, limit=10):
    """
    Get predictions for a specific user
    
    Args:
        user_id: User ID
        limit: Maximum number of records
    
    Returns:
        List of prediction records
    """
    # Import from firebase_config to use the correct database
    from firebase_config import get_patient_history
    
    try:
        # Use the firebase_config function which works with Realtime DB
        return get_patient_history(user_id=user_id, limit=limit)
    except Exception as e:
        print(f"❌ Error retrieving user predictions: {e}")
        return []

def get_user_statistics(user_id):
    """
    Get statistics for a specific user's predictions
    
    Args:
        user_id: User ID
    
    Returns:
        Dict with user statistics
    """
    # Import from firebase_config to use the correct database
    from firebase_config import get_statistics
    
    try:
        # Use the firebase_config function which works with Realtime DB
        return get_statistics(user_id=user_id)
    except Exception as e:
        print(f"❌ Error retrieving user statistics: {e}")
        return {
            'total_predictions': 0,
            'high_risk_count': 0,
            'low_risk_count': 0,
            'average_confidence': 0,
            'average_risk_percentage': 0
        }

    # OLD FIRESTORE CODE - keeping for reference but not used
    try:
        all_docs_old = db.collection('patient_predictions').where('user_id', '==', user_id).stream()
        
        total = 0
        high_risk = 0
        low_risk = 0
        
        for doc in all_docs:
            data = doc.to_dict()
            total += 1
            if data.get('risk_level') == 'high':
                high_risk += 1
            else:
                low_risk += 1
        
        stats = {
            'total_predictions': total,
            'high_risk_count': high_risk,
            'low_risk_count': low_risk,
            'high_risk_percentage': round((high_risk / total * 100) if total > 0 else 0, 2),
            'low_risk_percentage': round((low_risk / total * 100) if total > 0 else 0, 2)
        }
        
        return stats
        
    except Exception as e:
        print(f"❌ Error getting user statistics: {e}")
        return {
            'total_predictions': 0,
            'high_risk_count': 0,
            'low_risk_count': 0
        }

def change_password(user_id, old_password, new_password):
    """
    Change user password
    
    Args:
        user_id: User ID
        old_password: Current password (plain text)
        new_password: New password (plain text)
    
    Returns:
        Tuple (success: bool, message: str)
    """
    if not db:
        return False, "Database not initialized"
    
    try:
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return False, "User not found"
        
        user_data = user_doc.to_dict()
        
        # Verify old password
        if hash_password(old_password) != user_data.get('password_hash'):
            return False, "Incorrect current password"
        
        # Update password
        user_ref.update({
            'password_hash': hash_password(new_password)
        })
        
        return True, "Password changed successfully"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def update_user_profile(user_id, full_name=None, contact=None, address=None):
    """
    Update user profile information
    
    Args:
        user_id: User ID
        full_name: New full name (optional)
        contact: New contact (optional)
        address: New address (optional)
    
    Returns:
        Tuple (success: bool, message: str)
    """
    if not db:
        return False, "Database not initialized"
    
    try:
        user_ref = db.collection('users').document(user_id)
        
        updates = {}
        if full_name:
            updates['full_name'] = full_name
        if contact:
            updates['contact'] = contact
        if address:
            updates['address'] = address
        
        if updates:
            user_ref.update(updates)
            return True, "Profile updated successfully"
        else:
            return False, "No updates provided"
        
    except Exception as e:
        return False, f"Error: {str(e)}"
