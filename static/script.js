/**
 * Diabetes Health Predictor - AI Doctor Portal
 * JavaScript for Dynamic Form Handling & API Integration
 */

// ========================================
// GLOBAL VARIABLES
// ========================================
let currentPredictionData = null;
let currentReportFilename = null;

// ========================================
// DOM ELEMENTS
// ========================================
const patientForm = document.getElementById('patientForm');
const medicalForm = document.getElementById('medicalForm');
const registrationCard = document.getElementById('registrationCard');
const medicalTestCard = document.getElementById('medicalTestCard');
const resultsCard = document.getElementById('resultsCard');
const resultContainer = document.getElementById('resultContainer');
const reportModal = document.getElementById('reportModal');
const reportContent = document.getElementById('reportContent');
const reportFooter = document.getElementById('reportFooter');
const loadingOverlay = document.getElementById('loadingOverlay');
const predictBtn = document.getElementById('predictBtn');
const patientSummary = document.getElementById('patientSummary');

// Store patient data
let patientData = {};

// ========================================
// FORM VALIDATION & SUBMISSION
// ========================================

// Handle Patient Registration Form Submission
patientForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    // Collect patient data
    const name = document.getElementById('patientName').value.trim();
    const age = document.getElementById('patientAge').value;
    const sex = document.getElementById('patientSex').value;
    const contact = document.getElementById('patientContact').value.trim();
    const address = document.getElementById('patientAddress').value.trim();
    
    // Validate
    if (!name || !age || !sex || !contact) {
        showAlert('Please fill in all required fields!', 'error');
        return;
    }
    
    // Validate contact number
    if (contact.length !== 10 || !/^\d+$/.test(contact)) {
        showAlert('Please enter a valid 10-digit contact number!', 'error');
        return;
    }
    
    // Store patient data
    patientData = {
        name: name,
        age: parseInt(age),
        sex: sex,
        contact: contact,
        address: address
    };
    
    // Show patient summary in medical test card
    displayPatientSummary();
    
    // Hide registration, show medical test card
    registrationCard.style.display = 'none';
    medicalTestCard.style.display = 'block';
    medicalTestCard.scrollIntoView({ behavior: 'smooth' });
    
    showAlert('Patient registered successfully! Please enter health details.', 'success');
});

// Display patient summary
function displayPatientSummary() {
    patientSummary.innerHTML = `
        <h3><i class="fas fa-user-check"></i> Patient Information</h3>
        <div class="patient-summary-grid">
            <div class="patient-summary-item">
                <strong>Name:</strong>
                <span>${patientData.name}</span>
            </div>
            <div class="patient-summary-item">
                <strong>Age:</strong>
                <span>${patientData.age} years</span>
            </div>
            <div class="patient-summary-item">
                <strong>Sex:</strong>
                <span>${patientData.sex}</span>
            </div>
            <div class="patient-summary-item">
                <strong>Contact:</strong>
                <span>${patientData.contact}</span>
            </div>
        </div>
    `;
}

// Go back to registration
function goBackToRegistration() {
    medicalTestCard.style.display = 'none';
    registrationCard.style.display = 'block';
    registrationCard.scrollIntoView({ behavior: 'smooth' });
}

// Handle Medical Form Submission
medicalForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Collect all form data
    const formData = {
        // Patient Information
        ...patientData,
        
        // Medical Test Parameters
        pregnancies: parseFloat(document.getElementById('pregnancies').value) || 0,
        glucose: parseFloat(document.getElementById('glucose').value),
        bloodPressure: parseFloat(document.getElementById('bloodPressure').value),
        skinThickness: parseFloat(document.getElementById('skinThickness').value) || 20,
        insulin: parseFloat(document.getElementById('insulin').value) || 79,
        bmi: parseFloat(document.getElementById('bmi').value),
        diabetesPedigreeFunction: parseFloat(document.getElementById('diabetesPedigreeFunction').value),
    };
    
    // Validate required medical fields
    if (!formData.glucose || !formData.bloodPressure || !formData.bmi || !formData.diabetesPedigreeFunction) {
        showAlert('Please fill in all required medical test parameters marked with *', 'error');
        return;
    }
    
    // Make prediction
    await makePrediction(formData);
});

// ========================================
// API CALLS
// ========================================

/**
 * Make prediction API call
 */
async function makePrediction(data) {
    showLoading(true);
    predictBtn.disabled = true;
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentPredictionData = result;
            displayResults(result);
            resultsCard.style.display = 'block';
            resultsCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            // Play sound based on risk level
            playPredictionSound(result.risk_level);
        } else {
            showAlert(`Prediction Error: ${result.error}`, 'error');
        }
        
    } catch (error) {
        console.error('Prediction error:', error);
        showAlert('Failed to connect to the server. Please try again.', 'error');
    } finally {
        showLoading(false);
        predictBtn.disabled = false;
    }
}

/**
 * Generate AI doctor report
 */
async function generateReport() {
    if (!currentPredictionData) {
        showAlert('No prediction data available. Please run prediction first.', 'error');
        return;
    }
    
    // Show modal with loading
    reportModal.classList.add('show');
    reportContent.innerHTML = `
        <div class="loading">
            <i class="fas fa-spinner fa-spin"></i>
            <p>Generating comprehensive medical report...</p>
            <p style="font-size: 0.9rem; color: var(--text-gray); margin-top: 1rem;">
                This may take a few moments
            </p>
        </div>
    `;
    reportFooter.style.display = 'none';
    
    try {
        const response = await fetch('/report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(currentPredictionData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentReportFilename = result.report_file;
            displayReport(result.report);
            reportFooter.style.display = 'flex';
        } else {
            reportContent.innerHTML = `
                <div style="text-align: center; padding: 2rem; color: var(--danger-color);">
                    <i class="fas fa-exclamation-triangle" style="font-size: 3rem; margin-bottom: 1rem;"></i>
                    <p><strong>Report Generation Failed</strong></p>
                    <p>${result.error}</p>
                </div>
            `;
        }
        
    } catch (error) {
        console.error('Report generation error:', error);
        reportContent.innerHTML = `
            <div style="text-align: center; padding: 2rem; color: var(--danger-color);">
                <i class="fas fa-exclamation-triangle" style="font-size: 3rem; margin-bottom: 1rem;"></i>
                <p><strong>Connection Error</strong></p>
                <p>Failed to connect to the AI report generator.</p>
            </div>
        `;
    }
}

/**
 * Download the generated report
 */
function downloadReport() {
    if (currentReportFilename) {
        window.location.href = `/download_report/${currentReportFilename}`;
        showAlert('Report downloaded successfully!', 'success');
    } else {
        showAlert('No report file available to download.', 'error');
    }
}

// ========================================
// UI DISPLAY FUNCTIONS
// ========================================

/**
 * Display prediction results
 */
function displayResults(data) {
    const riskClass = data.risk_level === 'high' ? 'high-risk' : 'low-risk';
    const icon = data.risk_level === 'high' 
        ? '<i class="fas fa-exclamation-circle"></i>' 
        : '<i class="fas fa-check-circle"></i>';
    
    resultContainer.innerHTML = `
        <div class="result-badge ${riskClass}">
            ${icon} ${data.prediction}
        </div>
        
        <div class="confidence-badge">
            Confidence: ${data.confidence}%
        </div>
        
        <div class="result-details">
            <h3 style="margin-bottom: 1rem; color: var(--text-dark);">
                <i class="fas fa-user"></i> Patient Information
            </h3>
            <div class="result-row">
                <span><strong>Name:</strong></span>
                <span>${data.patient_info.name}</span>
            </div>
            <div class="result-row">
                <span><strong>Age:</strong></span>
                <span>${data.patient_info.age} years</span>
            </div>
            <div class="result-row">
                <span><strong>Sex:</strong></span>
                <span>${data.patient_info.sex}</span>
            </div>
            <div class="result-row">
                <span><strong>Contact:</strong></span>
                <span>${data.patient_info.contact}</span>
            </div>
        </div>
        
        <div class="result-details" style="margin-top: 1rem;">
            <h3 style="margin-bottom: 1rem; color: var(--text-dark);">
                <i class="fas fa-clipboard-list"></i> Medical Test Results
            </h3>
            ${Object.entries(data.medical_data).map(([key, value]) => `
                <div class="result-row">
                    <span><strong>${key}:</strong></span>
                    <span>${typeof value === 'number' ? value.toFixed(2) : value}</span>
                </div>
            `).join('')}
        </div>
        
        <div class="recommendation">
            <strong><i class="fas fa-lightbulb"></i> Recommendation:</strong><br>
            ${data.recommendation}
        </div>
    `;
}

/**
 * Display AI-generated report in modal
 */
function displayReport(reportText) {
    reportContent.innerHTML = `
        <div class="report-content">${escapeHtml(reportText)}</div>
    `;
}

/**
 * Show/hide loading overlay
 */
function showLoading(show) {
    if (show) {
        loadingOverlay.classList.add('show');
    } else {
        loadingOverlay.classList.remove('show');
    }
}

/**
 * Show alert message
 */
function showAlert(message, type = 'info') {
    const alertColors = {
        success: '#10b981',
        error: '#ef4444',
        warning: '#f59e0b',
        info: '#3b82f6'
    };
    
    const alertDiv = document.createElement('div');
    alertDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${alertColors[type]};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        z-index: 4000;
        animation: slideInRight 0.3s ease;
        max-width: 400px;
        font-weight: 500;
    `;
    alertDiv.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'times-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    
    document.body.appendChild(alertDiv);
    
    setTimeout(() => {
        alertDiv.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => alertDiv.remove(), 300);
    }, 4000);
}

// ========================================
// MODAL FUNCTIONS
// ========================================

/**
 * Close the report modal
 */
function closeModal() {
    reportModal.classList.remove('show');
}

// Close modal when clicking outside
reportModal.addEventListener('click', (e) => {
    if (e.target === reportModal) {
        closeModal();
    }
});

// Close modal with Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && reportModal.classList.contains('show')) {
        closeModal();
    }
});

// ========================================
// UTILITY FUNCTIONS
// ========================================

/**
 * Reset all forms
 */
function resetForms() {
    if (confirm('Are you sure you want to start a new assessment?')) {
        patientForm.reset();
        medicalForm.reset();
        registrationCard.style.display = 'block';
        medicalTestCard.style.display = 'none';
        resultsCard.style.display = 'none';
        currentPredictionData = null;
        currentReportFilename = null;
        patientData = {};
        window.scrollTo({ top: 0, behavior: 'smooth' });
        showAlert('Starting new patient assessment', 'info');
    }
}

/**
 * Start a new prediction (reset and scroll to top)
 */
function newPrediction() {
    resetForms();
}

/**
 * Play sound alert based on prediction result
 */
function playPredictionSound(riskLevel) {
    try {
        if (riskLevel === 'high') {
            // Play alert sound for high risk (diabetes detected)
            const alertSound = document.getElementById('alertSound');
            if (alertSound) {
                alertSound.volume = 0.5;
                alertSound.play().catch(err => console.log('Audio play prevented:', err));
            }
            
            // Show visual alert
            showAlert('⚠️ HIGH RISK DETECTED! Please consult a doctor immediately.', 'warning');
        } else {
            // Play success sound for low risk (safe, no diabetes)
            const successSound = document.getElementById('successSound');
            if (successSound) {
                successSound.volume = 0.5;
                successSound.play().catch(err => console.log('Audio play prevented:', err));
            }
            
            // Show visual success message
            showAlert('✅ Good News! Low risk detected. Maintain healthy lifestyle.', 'success');
        }
    } catch (error) {
        console.log('Sound playback error:', error);
    }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Validate contact number in real-time
 */
document.getElementById('patientContact').addEventListener('input', function(e) {
    this.value = this.value.replace(/\D/g, '');
    if (this.value.length > 10) {
        this.value = this.value.slice(0, 10);
    }
});

// ========================================
// ANIMATIONS FOR CSS
// ========================================

// Add slide animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ========================================
// INITIALIZATION
// ========================================

console.log('🏥 Diabetes Health Predictor - AI Doctor Portal initialized');
console.log('✅ All systems ready');
