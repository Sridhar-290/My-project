document.addEventListener('DOMContentLoaded', () => {
    const ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    const verdictText = document.getElementById('verdict-text');
    const verdictCard = document.getElementById('verdict-card');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceValue = document.getElementById('confidence-value');
    
    const metricLiveness = document.getElementById('metric-liveness');
    const metricTexture = document.getElementById('metric-texture');
    const metricStability = document.getElementById('metric-stability');
    
    const auditLog = document.getElementById('audit-log');

    let lastVerdict = "";

    function addLog(message, type = 'system') {
        const entry = document.createElement('p');
        const timestamp = new Date().toLocaleTimeString('en-GB', { hour12: false });
        entry.className = `log-entry ${type}`;
        entry.textContent = `[${timestamp}] ${message}`;
        auditLog.prepend(entry);
        
        // Keep only last 50 entries
        while (auditLog.children.length > 50) {
            auditLog.removeChild(auditLog.lastChild);
        }
    }

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Update Verdict
        verdictText.textContent = data.verdict;
        confidenceFill.style.width = `${data.confidence * 100}%`;
        confidenceValue.textContent = `${Math.round(data.confidence * 100)}%`;
        
        // Update Card Class
        verdictCard.className = 'verdict-container';
        if (data.verdict === 'REAL') verdictCard.classList.add('verdict-real');
        else if (data.verdict === 'SUSPICIOUS') verdictCard.classList.add('verdict-suspicious');
        else if (data.verdict === 'FAKE') verdictCard.classList.add('verdict-fake');

        // Update Metrics
        metricLiveness.style.width = `${data.indicators.liveness * 100}%`;
        metricTexture.style.width = `${data.indicators.texture * 100}%`;
        metricStability.style.width = `${data.indicators.stability * 100}%`;

        // Log significant changes
        if (data.verdict !== lastVerdict && data.verdict !== "Processing...") {
            const type = data.verdict === 'FAKE' ? 'alert' : 'system';
            addLog(`Verdict updated: ${data.verdict}`, type);
            lastVerdict = data.verdict;
        }

        if (data.blink_count > 0 && data.indicators.liveness > 0.5) {
            // Randomly log blink detection for activity feel
            if (Math.random() > 0.95) addLog("Biometric liveness confirmed: Blink detected.");
        }
    };

    ws.onclose = () => {
        addLog("Security connection lost.", "alert");
    };

    ws.onerror = () => {
        addLog("Security protocol error.", "alert");
    };
});
