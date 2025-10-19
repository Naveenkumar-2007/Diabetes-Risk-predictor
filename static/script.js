/**
 * Diabetes Health Predictor - AI Doctor Portal
 * JavaScript for Dynamic Form Handling & API Integration
 */

// ========================================
// GLOBAL VARIABLES
// ========================================
let currentPredictionData = null;
let currentReportFilename = null;
let currentComparisonAnalysis = null;
let comparisonInitialized = false;

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
const reportTabPane = document.getElementById('reportTabPane');
const comparisonTabPane = document.getElementById('comparisonTabPane');
const modalTabButtons = document.querySelectorAll('.modal-tab');
const pastPredictionsSelect = document.getElementById('pastPredictionsSelect');
const comparePredictionsBtn = document.getElementById('comparePredictionsBtn');
const comparisonStatus = document.getElementById('comparisonStatus');
const currentVsNormalImg = document.getElementById('currentVsNormalImg');
const comparisonGraphImg = document.getElementById('comparisonGraphImg');
const historicalGraphCard = document.getElementById('historicalGraphCard');
const groqExplanationCard = document.getElementById('groqExplanationCard');
const groqExplanation = document.getElementById('groqExplanation');
const downloadComparisonPdfBtn = document.getElementById('downloadComparisonPdf');

const MAX_COMPARISON_SELECTION = 3;

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
            resetComparisonUI();
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
    setReportModalTab('report');
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
            if (reportFooter) {
                const activeTab = document.querySelector('.modal-tab.active');
                const comparisonActive = activeTab && activeTab.dataset.tab === 'comparison';
                if (comparisonActive) {
                    reportFooter.dataset.restoreDisplay = 'flex';
                    reportFooter.style.display = 'none';
                } else {
                    reportFooter.style.display = 'flex';
                    delete reportFooter.dataset.restoreDisplay;
                }
            }
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

function resetComparisonUI() {
    comparisonInitialized = false;
    currentComparisonAnalysis = null;
    if (pastPredictionsSelect) {
        pastPredictionsSelect.innerHTML = '';
    }
    if (comparisonStatus) {
        comparisonStatus.textContent = '';
    }
    if (comparePredictionsBtn) {
        comparePredictionsBtn.disabled = true;
    }
    if (currentVsNormalImg) {
        currentVsNormalImg.removeAttribute('src');
    }
    if (comparisonGraphImg) {
        comparisonGraphImg.removeAttribute('src');
    }
    if (historicalGraphCard) {
        historicalGraphCard.style.display = 'none';
    }
    if (groqExplanation) {
        groqExplanation.innerHTML = '';
    }
    if (groqExplanationCard) {
        groqExplanationCard.style.display = 'none';
    }
    if (downloadComparisonPdfBtn) {
        downloadComparisonPdfBtn.style.display = 'none';
        downloadComparisonPdfBtn.dataset.downloadUrl = '';
    }
}

function updateCurrentVsNormalImage() {
    if (!currentVsNormalImg) {
        return;
    }

    const graphUrl = (currentComparisonAnalysis && currentComparisonAnalysis.current_vs_normal_graph_url)
        || currentPredictionData?.graphs?.current_vs_normal?.url
        || currentPredictionData?.current_vs_normal_graph_url;

    if (graphUrl) {
        currentVsNormalImg.src = appendCacheBuster(graphUrl);
    } else {
        currentVsNormalImg.removeAttribute('src');
    }
}

async function prepareComparisonTab(forceReload = false) {
    if (!comparisonTabPane) {
        return;
    }

    if (!currentPredictionData || !currentPredictionData.firebase_id) {
        resetComparisonUI();
        if (comparisonStatus) {
            comparisonStatus.textContent = 'Run a prediction before using trend comparison.';
        }
        return;
    }

    updateCurrentVsNormalImage();

    if (!comparisonInitialized || forceReload) {
        if (comparisonStatus) {
            comparisonStatus.textContent = 'Loading history...';
        }
        if (comparePredictionsBtn) {
            comparePredictionsBtn.disabled = true;
        }
        await populatePastPredictions(currentPredictionData.firebase_id);
        comparisonInitialized = true;
    } else if (comparisonStatus && !comparisonStatus.textContent) {
        comparisonStatus.textContent = 'Select two or three predictions for trend analysis.';
    }

    if (currentComparisonAnalysis) {
        renderComparisonAnalysis(currentComparisonAnalysis);
    }
}

function setReportModalTab(targetTab) {
    if (!modalTabButtons || modalTabButtons.length === 0) {
        return;
    }

    modalTabButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === targetTab);
    });

    if (targetTab === 'comparison') {
        if (reportTabPane) {
            reportTabPane.style.display = 'none';
        }
        if (comparisonTabPane) {
            comparisonTabPane.style.display = 'block';
        }
        if (reportFooter && !reportFooter.dataset.restoreDisplay) {
            reportFooter.dataset.restoreDisplay = reportFooter.style.display || '';
        }
        if (reportFooter) {
            reportFooter.style.display = 'none';
        }
        prepareComparisonTab().catch(err => console.error('Comparison prep error:', err));
    } else {
        if (reportTabPane) {
            reportTabPane.style.display = 'block';
        }
        if (comparisonTabPane) {
            comparisonTabPane.style.display = 'none';
        }
        if (reportFooter && reportFooter.dataset.restoreDisplay !== undefined) {
            reportFooter.style.display = reportFooter.dataset.restoreDisplay;
            delete reportFooter.dataset.restoreDisplay;
        }
    }
}

async function populatePastPredictions(currentPredictionId) {
    if (!pastPredictionsSelect) {
        return;
    }

    pastPredictionsSelect.innerHTML = '';
    if (comparisonStatus) {
        comparisonStatus.textContent = 'Loading history...';
    }

    try {
        const response = await fetch('/patient_history?limit=25');
        const data = await response.json();

        if (!data.success || !Array.isArray(data.history)) {
            if (comparisonStatus) {
                comparisonStatus.textContent = 'Unable to load prediction history.';
            }
            return;
        }

        const filteredHistory = data.history.filter(item => item.id && item.id !== currentPredictionId);
        filteredHistory.sort((a, b) => parseHistoryDate(a) - parseHistoryDate(b));
        if (filteredHistory.length < 2) {
            if (comparisonStatus) {
                comparisonStatus.textContent = 'Need at least two past predictions to run a comparison.';
            }
            if (comparePredictionsBtn) {
                comparePredictionsBtn.disabled = true;
            }
            return;
        }

        filteredHistory.forEach(item => {
            const option = document.createElement('option');
            option.value = item.id;
            option.textContent = formatPredictionOption(item);
            pastPredictionsSelect.appendChild(option);
        });

        if (comparisonStatus) {
            comparisonStatus.textContent = 'Select two or three predictions for trend analysis.';
        }
        if (comparePredictionsBtn) {
            comparePredictionsBtn.disabled = false;
        }

    } catch (error) {
        console.error('History fetch error:', error);
        if (comparisonStatus) {
            comparisonStatus.textContent = 'Failed to load prediction history.';
        }
        if (comparePredictionsBtn) {
            comparePredictionsBtn.disabled = true;
        }
    }
}

function parseHistoryDate(item) {
    if (!item) {
        return 0;
    }

    if (item.timestamp) {
        const date = new Date(item.timestamp);
        if (!Number.isNaN(date.getTime())) {
            return date.getTime();
        }
    }

    if (item.created_at) {
        const created = Number(item.created_at);
        if (!Number.isNaN(created)) {
            return created * 1000;
        }
    }

    if (item.date) {
        const formatted = `${item.date}T${item.time || '00:00:00'}`;
        const date = new Date(formatted);
        if (!Number.isNaN(date.getTime())) {
            return date.getTime();
        }
    }

    return 0;
}

function formatPredictionOption(item) {
    const dateValue = parseHistoryDate(item);
    const dateObj = new Date(dateValue);
    const hasValidDate = !Number.isNaN(dateObj.getTime());
    const dateLabel = hasValidDate
        ? `${dateObj.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })} ${dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
        : 'Previous Visit';

    const resultText = item.result || item.prediction || item.risk_level || 'Unknown Risk';
    const confidenceValue = item.confidence !== undefined && item.confidence !== null
        ? Number(item.confidence).toFixed(1)
        : null;
    const confidenceText = confidenceValue ? ` (${confidenceValue}%)` : '';

    return `${dateLabel} - ${resultText}${confidenceText}`;
}

function appendCacheBuster(url) {
    if (!url) {
        return '';
    }
    const separator = url.includes('?') ? '&' : '?';
    return `${url}${separator}t=${Date.now()}`;
}

function formatExplanation(text) {
    if (!text) {
        return '<p>No explanation available.</p>';
    }

    return text
        .split('\n')
        .map(line => line.trim())
        .filter(Boolean)
        .map(line => `<p>${escapeHtml(line)}</p>`)
        .join('');
}

function renderComparisonAnalysis(result) {
    if (!result) {
        if (comparisonGraphImg) {
            comparisonGraphImg.removeAttribute('src');
        }
        if (historicalGraphCard) {
            historicalGraphCard.style.display = 'none';
        }
        if (groqExplanation) {
            groqExplanation.innerHTML = '';
        }
        if (groqExplanationCard) {
            groqExplanationCard.style.display = 'none';
        }
        if (downloadComparisonPdfBtn) {
            downloadComparisonPdfBtn.style.display = 'none';
            downloadComparisonPdfBtn.dataset.downloadUrl = '';
        }
        return;
    }

    const comparisonGraphUrl = result.comparison_graph_url;
    if (comparisonGraphImg) {
        if (comparisonGraphUrl) {
            comparisonGraphImg.src = appendCacheBuster(comparisonGraphUrl);
            if (historicalGraphCard) {
                historicalGraphCard.style.display = 'block';
            }
        } else {
            comparisonGraphImg.removeAttribute('src');
            if (historicalGraphCard) {
                historicalGraphCard.style.display = 'none';
            }
        }
    }

    if (groqExplanation) {
        groqExplanation.innerHTML = formatExplanation(result.explanation);
    }
    if (groqExplanationCard) {
        groqExplanationCard.style.display = 'block';
    }

    if (downloadComparisonPdfBtn) {
        if (result.report_download_url) {
            downloadComparisonPdfBtn.dataset.downloadUrl = result.report_download_url;
            downloadComparisonPdfBtn.style.display = 'inline-flex';
        } else {
            downloadComparisonPdfBtn.style.display = 'none';
            downloadComparisonPdfBtn.dataset.downloadUrl = '';
        }
    }
}

async function compareWithPast() {
    if (!pastPredictionsSelect || !currentPredictionData || !currentPredictionData.firebase_id) {
        showAlert('Run a prediction before requesting comparisons.', 'warning');
        return;
    }

    const selected = Array.from(pastPredictionsSelect.selectedOptions).map(option => option.value);

    if (selected.length < 2 || selected.length > MAX_COMPARISON_SELECTION) {
        showAlert(`Please select between 2 and ${MAX_COMPARISON_SELECTION} past predictions.`, 'warning');
        return;
    }

    if (comparisonStatus) {
        comparisonStatus.textContent = 'Generating comparison analysis...';
    }
    if (comparePredictionsBtn) {
        comparePredictionsBtn.disabled = true;
    }

    try {
        const response = await fetch('/prediction/analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                current_prediction_id: currentPredictionData.firebase_id,
                past_prediction_ids: selected
            })
        });

        const result = await response.json();

        if (!result.success) {
            showAlert(`Comparison Error: ${result.error}`, 'error');
            if (comparisonStatus) {
                comparisonStatus.textContent = result.error || 'Comparison failed.';
            }
            return;
        }

        currentComparisonAnalysis = result;
        updateCurrentVsNormalImage();
        renderComparisonAnalysis(result);

        if (comparisonStatus) {
            comparisonStatus.textContent = 'Comparison updated successfully.';
        }

    } catch (error) {
        console.error('Comparison analysis error:', error);
        showAlert('Failed to analyze past predictions. Please try again later.', 'error');
        if (comparisonStatus) {
            comparisonStatus.textContent = 'Comparison failed. Try again later.';
        }
    } finally {
        if (comparePredictionsBtn) {
            comparePredictionsBtn.disabled = false;
        }
    }
}

function downloadComparisonReport() {
    if (!downloadComparisonPdfBtn) {
        return;
    }

    const downloadUrl = downloadComparisonPdfBtn.dataset.downloadUrl;

    if (!downloadUrl) {
        showAlert('Generate a comparison before downloading the report.', 'warning');
        return;
    }

    window.location.href = downloadUrl;
    showAlert('Preparing comparison report download...', 'info');
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
    setReportModalTab('report');
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
        currentComparisonAnalysis = null;
        comparisonInitialized = false;
        patientData = {};
        resetComparisonUI();

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

if (pastPredictionsSelect) {
    pastPredictionsSelect.addEventListener('change', () => {
        const selected = Array.from(pastPredictionsSelect.selectedOptions);
        if (selected.length > MAX_COMPARISON_SELECTION) {
            selected[selected.length - 1].selected = false;
            showAlert(`You can compare up to ${MAX_COMPARISON_SELECTION} predictions at a time.`, 'warning');
        }
    });
}

if (modalTabButtons && modalTabButtons.length > 0) {
    modalTabButtons.forEach(btn => {
        btn.addEventListener('click', () => setReportModalTab(btn.dataset.tab));
    });
}

if (comparePredictionsBtn) {
    comparePredictionsBtn.addEventListener('click', compareWithPast);
}

if (downloadComparisonPdfBtn) {
    downloadComparisonPdfBtn.addEventListener('click', downloadComparisonReport);
}

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
