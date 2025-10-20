"""
User Authentication and Authorization Module
Handles user registration, login, and role-based access control
"""
import os
import re
import hashlib
import secrets
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from functools import wraps
from flask import session, redirect, url_for, request
from firebase_config import db

try:
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False

# Admin credentials (hardcoded for security)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_HASH = "e54fc6b51915e222ba6196747a19ebb8dfa651fd2b46a385a0ded647fbfefda0"  # Change this password!

# OAuth and email configuration
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
PASSWORD_RESET_EXPIRY_HOURS = int(os.getenv('PASSWORD_RESET_EXPIRY_HOURS', '1'))

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_session_token():
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)


def _generate_username_from_email(email):
    """Create a unique username based on the email address"""
    base_part = (email or '').split('@')[0]
    sanitized = re.sub(r'[^A-Za-z0-9_]', '', base_part) or 'user'
    candidate = sanitized.lower()

    if not db:
        return candidate

    try:
        users_ref = db.collection('users')
    except Exception:
        return candidate

    suffix = 0
    while True:
        check_username = candidate if suffix == 0 else f"{candidate}{suffix}"
        try:
            existing = users_ref.where('username', '==', check_username).limit(1).stream()
            if len(list(existing)) == 0:
                return check_username
        except Exception:
            return check_username
        suffix += 1


def send_password_reset_email(recipient_email, reset_url):
    """Send password reset instructions via SMTP"""
    smtp_host = os.getenv('SMTP_HOST')
    if not smtp_host:
        print(f"⚠️ SMTP_HOST not configured. Password reset link for {recipient_email}: {reset_url}")
        return False, "Email service is not configured. Please contact the administrator."

    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    smtp_username = os.getenv('SMTP_USERNAME')
    smtp_password = os.getenv('SMTP_PASSWORD')
    use_tls = os.getenv('SMTP_USE_TLS', 'true').lower() in ('true', '1', 'yes')
    
    # Parse sender email properly
    sender_email_raw = os.getenv('SMTP_FROM_EMAIL') or smtp_username or 'no-reply@diabetes-predictor.local'
    # Extract just the email if it's in format "Name <email@example.com>"
    if '<' in sender_email_raw and '>' in sender_email_raw:
        sender_email = sender_email_raw.split('<')[1].split('>')[0].strip()
        sender_name = sender_email_raw.split('<')[0].strip()
    else:
        sender_email = sender_email_raw.strip()
        sender_name = sender_email_raw.strip()

    subject = 'Reset your Diabetes Health Predictor password'
    body = (
        "We received a request to reset your Diabetes Health Predictor account password.\n\n"
        f"If you made this request, click the link below within {PASSWORD_RESET_EXPIRY_HOURS} hour(s) to choose a new password:\n\n"
        f"{reset_url}\n\n"
        "If you did not request a reset, you can safely ignore this email.\n\n"
        "Best regards,\n"
        "Diabetes Health Predictor Team"
    )

    message = MIMEText(body)
    message['Subject'] = subject
    message['From'] = sender_email_raw  # Use the full format with name
    message['To'] = recipient_email

    try:
        print(f"📧 Attempting to send password reset email to {recipient_email}")
        print(f"   SMTP Host: {smtp_host}:{smtp_port}")
        print(f"   From: {sender_email_raw}")
        print(f"   Login User: {smtp_username}")
        
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.set_debuglevel(1)  # Enable debug output
            if use_tls:
                print("   Starting TLS...")
                server.starttls()
            if smtp_username and smtp_password:
                print(f"   Logging in as: {smtp_username}")
                server.login(smtp_username, smtp_password)
            print("   Sending email...")
            server.sendmail(sender_email, [recipient_email], message.as_string())
        
        print(f"✅ Password reset email sent successfully to {recipient_email}")
        return True, "Password reset email sent successfully"
    except smtplib.SMTPAuthenticationError as exc:
        print(f"❌ SMTP Authentication failed: {exc}")
        print(f"   Username: {smtp_username}")
        print(f"   Password length: {len(smtp_password) if smtp_password else 0}")
        return False, "Email authentication failed. Please check SMTP credentials."
    except smtplib.SMTPException as exc:
        print(f"❌ SMTP error sending password reset email: {exc}")
        return False, f"Unable to send email: {str(exc)}"
    except Exception as exc:
        print(f"❌ Unexpected error sending password reset email: {exc}")
        import traceback
        traceback.print_exc()
        return False, f"Email delivery failed: {str(exc)}"

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


def initiate_password_reset(email, base_url):
    """Start the password reset flow for the provided email"""
    if not db:
        return False, "Database not initialized"

    normalized_email = (email or '').strip().lower()
    if not normalized_email:
        return False, "Email is required"

    try:
        users_ref = db.collection('users')
        user_matches = list(users_ref.where('email', '==', normalized_email).limit(1).stream())
    except Exception as exc:
        print(f"❌ Password reset lookup error: {exc}")
        return False, "Unable to process password reset right now"

    if not user_matches:
        return True, "If an account exists for that email, we've sent reset instructions"

    user_doc = user_matches[0]
    user_data = user_doc.to_dict()
    if not user_data.get('is_active', True):
        return True, "If an account exists for that email, we've sent reset instructions"

    if user_data.get('auth_provider') == 'google' and not user_data.get('password_hash'):
        return True, "This account uses Google Sign-In. Please continue with Google to access your dashboard"

    token = secrets.token_urlsafe(48)
    expires_at = datetime.utcnow() + timedelta(hours=PASSWORD_RESET_EXPIRY_HOURS)

    try:
        tokens_ref = db.collection('password_resets')
        tokens_ref.document(token).set({
            'token': token,
            'user_id': user_doc.id,
            'email': normalized_email,
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': expires_at.isoformat(),
            'used': False
        })
    except Exception as exc:
        print(f"❌ Unable to persist password reset token: {exc}")
        return False, "Unable to process password reset right now"

    base = (base_url or '').rstrip('/')
    reset_url = f"{base}/reset-password?token={token}" if base else f"/reset-password?token={token}"

    email_sent, email_message = send_password_reset_email(normalized_email, reset_url)
    if not email_sent:
        print(f"⚠️ Password reset delivery failed for {normalized_email}: {email_message}")
        return False, email_message or "Unable to send password reset email right now"

    return True, "If an account exists for that email, we've sent reset instructions"


def validate_password_reset_token(token):
    """Check whether a password reset token is valid"""
    if not db:
        return False, "Database not initialized", None

    token_value = (token or '').strip()
    if not token_value:
        return False, "Invalid or expired reset link", None

    try:
        tokens_ref = db.collection('password_resets')
        token_doc = tokens_ref.document(token_value)
        token_snapshot = token_doc.get()
    except Exception as exc:
        print(f"❌ Password reset token lookup error: {exc}")
        return False, "Invalid or expired reset link", None

    if not getattr(token_doc, 'exists', False):
        return False, "Invalid or expired reset link", None

    token_data = token_snapshot.to_dict()
    if token_data.get('used'):
        return False, "This reset link has already been used", None

    expires_at = token_data.get('expires_at')
    try:
        expiry_dt = datetime.fromisoformat(expires_at) if expires_at else None
    except Exception:
        expiry_dt = None

    if not expiry_dt or expiry_dt < datetime.utcnow():
        try:
            token_doc.delete()
        except Exception:
            pass
        return False, "This reset link has expired", None

    return True, "Valid token", token_data


def reset_password_with_token(token, new_password):
    """Update the user's password when a valid token is provided"""
    token_value = (token or '').strip()

    if len((new_password or '').strip()) < 6:
        return False, "Password must be at least 6 characters"

    valid, message, token_data = validate_password_reset_token(token_value)
    if not valid:
        return False, message

    user_id = token_data.get('user_id')
    if not user_id:
        return False, "User account not found"

    try:
        users_ref = db.collection('users')
        user_doc = users_ref.document(user_id)
    except Exception as exc:
        print(f"❌ Password reset update error: {exc}")
        return False, "Unable to reset password right now"

    if not user_doc.exists:
        return False, "User account not found"

    try:
        user_doc.update({
            'password_hash': hash_password(new_password.strip()),
            'auth_provider': 'password',
            'last_password_reset': datetime.utcnow().isoformat()
        })
    except Exception as exc:
        print(f"❌ Failed to update password: {exc}")
        return False, "Unable to reset password right now"

    try:
        tokens_ref = db.collection('password_resets')
        tokens_ref.document(token_value).update({
            'used': True,
            'used_at': datetime.utcnow().isoformat()
        })
    except Exception:
        pass

    return True, "Password reset successfully"

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


def authenticate_google_user(id_token_str):
    """Authenticate or create users using Google Sign-In"""
    if not GOOGLE_AUTH_AVAILABLE:
        return False, "Google authentication library not available", None

    if not GOOGLE_CLIENT_ID:
        return False, "Google Sign-In is not configured", None

    credential = (id_token_str or '').strip()
    if not credential:
        return False, "Missing Google credential", None

    try:
        id_info = google_id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )
    except Exception as exc:
        print(f"❌ Google token verification failed: {exc}")
        return False, "Unable to verify Google credential", None

    email = (id_info.get('email') or '').lower()
    if not email:
        return False, "Google account is missing an email address", None

    full_name = id_info.get('name') or email.split('@')[0]
    picture = id_info.get('picture')
    google_subject = id_info.get('sub')

    if not db:
        return False, "Database not initialized", None

    try:
        users_ref = db.collection('users')
        matching_users = list(users_ref.where('email', '==', email).limit(1).stream())
    except Exception as exc:
        print(f"❌ Google login lookup error: {exc}")
        return False, "Unable to complete Google login", None

    if matching_users:
        user_doc = matching_users[0]
        user_data = user_doc.to_dict()
        if not user_data.get('is_active', True):
            return False, "Account is disabled", None

        updates = {
            'last_login': datetime.utcnow().isoformat(),
            'auth_provider': user_data.get('auth_provider', 'google'),
            'google_subject': google_subject,
            'email_verified': bool(id_info.get('email_verified'))
        }
        if picture:
            updates['picture'] = picture
        if full_name and not user_data.get('full_name'):
            updates['full_name'] = full_name
        if not user_data.get('username'):
            updates['username'] = _generate_username_from_email(email)

        try:
            user_doc.update(updates)
        except Exception as exc:
            print(f"⚠️ Failed to update Google user metadata: {exc}")

        user_data.update(updates)
        user_id = user_doc.id
    else:
        username = _generate_username_from_email(email)
        user_payload = {
            'username': username,
            'email': email,
            'full_name': full_name,
            'contact': '',
            'address': '',
            'role': 'user',
            'created_at': datetime.utcnow().isoformat(),
            'last_login': datetime.utcnow().isoformat(),
            'is_active': True,
            'auth_provider': 'google',
            'google_subject': google_subject,
            'email_verified': bool(id_info.get('email_verified')),
            'picture': picture,
            'password_hash': None
        }

        try:
            doc_ref = users_ref.add(user_payload)
            user_id = doc_ref[1].id
            user_data = user_payload
        except Exception as exc:
            print(f"❌ Failed to create Google user: {exc}")
            return False, "Unable to complete Google login", None

    user_info = {
        'user_id': user_id,
        'username': user_data.get('username') or _generate_username_from_email(email),
        'email': user_data.get('email'),
        'full_name': user_data.get('full_name') or full_name,
        'contact': user_data.get('contact', ''),
        'address': user_data.get('address', ''),
        'role': user_data.get('role', 'user'),
        'picture': user_data.get('picture', picture)
    }

    print(f"✅ Google user authenticated: {email}")
    return True, "Login successful", user_info


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
