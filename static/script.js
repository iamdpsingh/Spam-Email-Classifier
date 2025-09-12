// Sample email examples
const examples = {
    spam: "🎉 CONGRATULATIONS! You've won $10,000 CASH PRIZE! Click this link immediately to claim your prize before it expires in 24 hours! This is a LIMITED TIME OFFER! Don't miss out on this amazing opportunity! Call now: 1-800-WINNER! Free money guaranteed! Act fast or lose forever! URGENT! URGENT! URGENT!",
    ham: "Hi Sarah, I hope you're doing well. I wanted to follow up on our meeting yesterday regarding the quarterly marketing report. Could you please send me the updated sales figures by Friday afternoon? I need to review them before our presentation to the board next week. Let me know if you need any additional information from my end. Thanks for your help! Best regards, John"
};

// Fill textarea with example content
function fillExample(type) {
    const textarea = document.getElementById('email_text');
    if (textarea) {
        textarea.value = examples[type];
        textarea.focus();
        
        // Auto-resize textarea
        textarea.style.height = 'auto';
        textarea.style.height = Math.max(150, textarea.scrollHeight) + 'px';
        
        // Add visual feedback
        textarea.style.border = '2px solid #3498db';
        setTimeout(() => {
            textarea.style.border = '2px solid #e0e0e0';
        }, 1000);
    }
}

// Clear the form
function clearForm() {
    const textarea = document.getElementById('email_text');
    if (textarea) {
        textarea.value = '';
        textarea.style.height = '150px';
        textarea.focus();
    }
}

// Form validation and submission
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('emailForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            const textarea = document.getElementById('email_text');
            const text = textarea.value.trim();
            
            if (text.length < 10) {
                e.preventDefault();
                showNotification('Please enter at least 10 characters of email content for accurate classification.', 'warning');
                textarea.focus();
                return false;
            }
            
            if (text.length > 5000) {
                e.preventDefault();
                showNotification('Email content is too long. Please limit to 5000 characters.', 'warning');
                textarea.focus();
                return false;
            }
            
            // Show loading state
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '⏳ Classifying...';
                submitBtn.disabled = true;
            }
        });
    }
});

// Auto-resize textarea
document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.getElementById('email_text');
    if (textarea) {
        // Character counter
        const charCounter = document.createElement('div');
        charCounter.className = 'char-counter';
        charCounter.style.cssText = 'text-align: right; font-size: 0.9em; color: #666; margin-top: 5px;';
        textarea.parentNode.appendChild(charCounter);
        
        function updateCounter() {
            const count = textarea.value.length;
            charCounter.textContent = `${count}/5000 characters`;
            
            if (count > 5000) {
                charCounter.style.color = '#e74c3c';
            } else if (count > 4000) {
                charCounter.style.color = '#f39c12';
            } else {
                charCounter.style.color = '#666';
            }
        }
        
        textarea.addEventListener('input', function() {
            // Auto-resize
            this.style.height = 'auto';
            this.style.height = Math.max(150, this.scrollHeight) + 'px';
            
            // Update counter
            updateCounter();
        });
        
        // Initial counter update
        updateCounter();
    }
});

// Smooth scroll to results
function scrollToResults() {
    const resultContainer = document.querySelector('.result-container');
    if (resultContainer) {
        resultContainer.scrollIntoView({ 
            behavior: 'smooth',
            block: 'center'
        });
    }
}

// Call scroll function if results are present
window.addEventListener('load', function() {
    if (document.querySelector('.result-container')) {
        setTimeout(scrollToResults, 300);
    }
});

// Image modal functionality
document.addEventListener('DOMContentLoaded', function() {
    const images = document.querySelectorAll('.analysis-chart');
    
    images.forEach(img => {
        img.addEventListener('click', function() {
            createImageModal(this);
        });
    });
});

function createImageModal(img) {
    // Create modal elements
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.9);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
        cursor: pointer;
        opacity: 0;
        transition: opacity 0.3s ease;
    `;
    
    const modalImg = document.createElement('img');
    modalImg.src = img.src;
    modalImg.alt = img.alt;
    modalImg.style.cssText = `
        max-width: 95%;
        max-height: 95%;
        border-radius: 10px;
        box-shadow: 0 0 50px rgba(255,255,255,0.1);
    `;
    
    const closeBtn = document.createElement('div');
    closeBtn.innerHTML = '✕';
    closeBtn.style.cssText = `
        position: absolute;
        top: 20px;
        right: 30px;
        color: white;
        font-size: 30px;
        font-weight: bold;
        cursor: pointer;
        z-index: 1001;
    `;
    
    modal.appendChild(modalImg);
    modal.appendChild(closeBtn);
    document.body.appendChild(modal);
    
    // Show modal with animation
    setTimeout(() => {
        modal.style.opacity = '1';
    }, 10);
    
    // Close modal functionality
    function closeModal() {
        modal.style.opacity = '0';
        setTimeout(() => {
            document.body.removeChild(modal);
        }, 300);
    }
    
    modal.addEventListener('click', closeModal);
    closeBtn.addEventListener('click', closeModal);
    
    // Close on Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeModal();
        }
    });
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 1000;
        opacity: 0;
        transform: translateX(100%);
        transition: all 0.3s ease;
        max-width: 400px;
    `;
    
    // Set background color based on type
    const colors = {
        'info': '#3498db',
        'success': '#27ae60',
        'warning': '#f39c12',
        'error': '#e74c3c'
    };
    
    notification.style.backgroundColor = colors[type] || colors['info'];
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Show notification
    setTimeout(() => {
        notification.style.opacity = '1';
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Hide notification after 5 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, 5000);
}

// API usage example (for developers)
function classifyEmailViaAPI(emailText) {
    return fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: emailText })
    })
    .then(response => response.json())
    .then(data => {
        console.log('API Response:', data);
        return data;
    })
    .catch(error => {
        console.error('API Error:', error);
        return null;
    });
}

// Add loading animation for charts
document.addEventListener('DOMContentLoaded', function() {
    const charts = document.querySelectorAll('.analysis-chart');
    
    charts.forEach(chart => {
        chart.style.opacity = '0';
        chart.style.transform = 'translateY(20px)';
        chart.style.transition = 'all 0.6s ease';
        
        // Animate when image loads
        chart.addEventListener('load', function() {
            setTimeout(() => {
                this.style.opacity = '1';
                this.style.transform = 'translateY(0)';
            }, 200);
        });
        
        // If image is already cached
        if (chart.complete) {
            setTimeout(() => {
                chart.style.opacity = '1';
                chart.style.transform = 'translateY(0)';
            }, 200);
        }
    });
});
