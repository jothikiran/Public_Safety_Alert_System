document.addEventListener('DOMContentLoaded', function() {
    // Update gesture status in real-time
    function updateGestureStatus() {
        fetch('/gesture_status')
            .then(response => response.json())
            .then(data => {
                document.getElementById('gesture-status').textContent = data.gesture;
                setTimeout(updateGestureStatus, 1000);
            });
    }
    
    // Check for alarm status
    function checkAlarmStatus() {
        fetch('/alarm_status')
            .then(response => response.json())
            .then(data => {
                if (data.alarm) {
                    alert('EMERGENCY DETECTED! Alerting authorities...');
                }
                setTimeout(checkAlarmStatus, 2000);
            });
    }
    
    // Simulate noise level changes
    function simulateNoiseLevel() {
        const noiseElement = document.getElementById('noise-value');
        const noiseBar = document.querySelector('.noise-bar');
        let currentNoise = parseInt(noiseElement.textContent);
        
        // Random fluctuation between 55-75 dB
        const newNoise = Math.max(50, Math.min(75, currentNoise + (Math.random() * 6 - 3)));
        noiseElement.textContent = Math.round(newNoise);
        noiseBar.style.width = `calc(${newNoise}% * 0.65)`;
        
        setTimeout(simulateNoiseLevel, 3000);
    }
    
    // Initialize functions
    updateGestureStatus();
    checkAlarmStatus();
    simulateNoiseLevel();
});