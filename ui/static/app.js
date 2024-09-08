function renderReport(report) {
    let table = document.getElementById('report-table');
    let header = table.createTHead();
    let row = header.insertRow(0);

    // Create header
    let headers = ['Metric', 'Precision', 'Recall', 'F1-Score', 'Support'];
    headers.forEach((text, index) => {
        let th = document.createElement('th');
        th.innerText = text;
        row.appendChild(th);
    });

    // Fill data
    for (let label in report) {
        if (!['accuracy', 'macro avg', 'weighted avg'].includes(label)) {
            let row = table.insertRow();
            let labelCell = row.insertCell();
            labelCell.innerText = label;

            let precisionCell = row.insertCell();
            precisionCell.innerText = report[label]['precision'].toFixed(2);

            let recallCell = row.insertCell();
            recallCell.innerText = report[label]['recall'].toFixed(2);

            let f1Cell = row.insertCell();
            f1Cell.innerText = report[label]['f1-score'].toFixed(2);

            let supportCell = row.insertCell();
            supportCell.innerText = report[label]['support'];
        }
    }
}

function renderPieChart(pieData) {
    const ctx = document.getElementById('pieChart').getContext('2d');
    const labels = Object.keys(pieData);
    const data = Object.values(pieData);
    
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56'],
                hoverBackgroundColor: ['#FF6384', '#36A2EB', '#FFCE56']
            }]
        },
        options: {
            responsive: true
        }
    });
}
