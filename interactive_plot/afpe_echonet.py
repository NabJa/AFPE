import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch

from posenc.nets.positional_encodings import isotropic_fpe


def show_positional_similarity(grid_size, pos_embedding, idx, cosine=False):
    query = pos_embedding[idx, :].unsqueeze(0)
    similarity = torch.matmul(query, pos_embedding.T)
    
    if cosine:
        query_norm = torch.norm(query, p=2, dim=1)
        pos_norms = torch.norm(pos_embedding, p=2, dim=1)
        similarity = similarity / (query_norm * pos_norms)
    
    return similarity.view(*grid_size)

NUM_FRAMES = 5
GRID = [NUM_FRAMES, 16, 16]

S_TIME_VALUES = np.logspace(-1, 0.5, 7).round(2)
S_COL_VALUES =  np.logspace(-1, 0.5, 7).round(2)
S_ROW_VALUES =  np.logspace(-1, 0.5, 7).round(2)


# Pre-compute all data
all_data = {}
global_min = float('inf')
global_max = float('-inf')

for time_idx, s_time in enumerate(S_TIME_VALUES):
    for row_idx, s_row in enumerate(S_ROW_VALUES):
        for col_idx, s_col in enumerate(S_COL_VALUES):
            pe = isotropic_fpe(grid_size=GRID, hidden_size=256, spatial_dims=3, 
                            variance_factors=[s_time, s_row, s_col])[0]
            sim = show_positional_similarity(grid_size=GRID, pos_embedding=pe, idx=648)
            
            key = f"{time_idx}_{row_idx}_{col_idx}"
            all_data[key] = [sim[i].numpy().tolist() for i in range(NUM_FRAMES)]
            
            current_min = sim.min().item()
            current_max = sim.max().item()
            global_min = min(global_min, current_min)
            global_max = max(global_max, current_max)

# Create custom HTML with interactive sliders
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>AFPE</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            flex-direction: column; 
        }
        .sliders-wrapper {
            display: flex;
            gap: 20px; /* Adds spacing between sliders */
            justify-content: center;
            align-items: center;
            margin: 20px 0;
        }
        .slider-container { 
            display: flex;
            flex-direction: column; /* Stack label, slider, and value */
            align-items: center;
        }
        .slider-label, .slider-value { 
            margin-bottom: 5px;
        }

        #plotContainer { 
            width: 90%; 
            margin: 0 auto; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            flex-direction: column; 
            text-align: center; 
        }
    </style>
</head>
<body>
    <h1>Positional Similarity Across Time and Space</h1>
    <h2>Anisotropic Fourier Feature Positional Encoding</h2>
    <p>Explore the positional (dot product) similarity when alternating the hyperarameters described in the paper.</p>

    <div class="sliders-wrapper">
        <div class="slider-container">
            <span class="slider-label">s<sub>time</sub></span>
            <span class="slider-value" id="timeValue">0.6</span>
            <input type="range" id="timeSlider" min="0" max="6" value="3" step="1">
        </div>

        <div class="slider-container">
            <span class="slider-label">s<sub>row</sub></span>
            <span class="slider-value" id="rowValue">0.6</span>
            <input type="range" id="rowSlider" min="0" max="6" value="3" step="1">
        </div>

        <div class="slider-container">
            <span class="slider-label">s<sub>col</sub></span>
            <span class="slider-value" id="colValue">0.6</span>
            <input type="range" id="colSlider" min="0" max="6" value="3" step="1">
        </div>
    </div>

    <p>Shown is the similarity to the central patch (frame=2, x=8, y=8) in a grid of 5&#215;16&#215;16 patches.</p>
    
    <div id="plotContainer"></div>
    
    <script>
        // Data loaded from Python
        const allData = DATA_PLACEHOLDER;
        const timeValues = TIME_VALUES_PLACEHOLDER;
        const colValues = COL_VALUES_PLACEHOLDER;
        const rowValues = ROW_VALUES_PLACEHOLDER;
       
        const globalMin = GLOBAL_MIN_PLACEHOLDER;
        const globalMax = GLOBAL_MAX_PLACEHOLDER;
        const numFrames = NUM_FRAMES_PLACEHOLDER;
        
        const layout = {
            grid: {
                rows: 1,
                columns: numFrames,
                pattern: 'independent',
                xgap: 0.05  // Keep gaps consistent
            },
            height: 350,  
            width: numFrames * 250,  
            margin: {t: 50, b: 50, l: 50, r: 50},
            title: '',
            showlegend: false
        };

        // Fix aspect ratios and ensure uniform domains
        for (let i = 0; i < numFrames; i++) {
            console.log([i / numFrames, (i + 1) / numFrames])
            layout[`xaxis${i+1}`] = {
                title: `Frame ${i}`,
                domain: [i / numFrames, (i + 1) / numFrames],  // Ensure equal width
                constrain: 'domain'
            };
            layout[`yaxis${i+1}`] = {
                domain: [0, 1],  // Ensure full height consistency
                constrain: 'domain',
                showticklabels: (i === 0)
            };
        }
                
        // Initial data
        let currentTimeIdx = 3;
        let currentRowIdx =  3;
        let currentColIdx =  3;
        
        // Create initial plot
        function createPlot() {
            const data = [];
            const key = `${currentTimeIdx}_${currentRowIdx}_${currentColIdx}`;
            
            for (let i = 0; i < numFrames; i++) {
                data.push({
                    z: allData[key][i],
                    type: 'heatmap',
                    colorscale: 'Viridis',
                    zmin: globalMin,
                    zmax: globalMax,
                    showscale: (i === numFrames - 1),
                    xaxis: `x${i+1}`,
                    yaxis: `y${i+1}`
                });
            }
            
            Plotly.newPlot('plotContainer', data, layout);
        }
        
        // Update plot with new data
        function updatePlot() {
            const key = `${currentTimeIdx}_${currentRowIdx}_${currentColIdx}`;
            const update = {
                z: allData[key]
            };
            
            Plotly.update('plotContainer', update);
        }
        
        // Initialize the plot
        createPlot();
        
        // Time slider event
        document.getElementById('timeSlider').addEventListener('input', function() {
            currentTimeIdx = parseInt(this.value);
            document.getElementById('timeValue').textContent = timeValues[currentTimeIdx];
            updatePlot();
        });
        
        // Row slider event
        document.getElementById('rowSlider').addEventListener('input', function() {
            currentRowIdx = parseInt(this.value);
            document.getElementById('rowValue').textContent = rowValues[currentRowIdx];
            updatePlot();
        });

        // Col slider event
        document.getElementById('colSlider').addEventListener('input', function() {
            currentColIdx = parseInt(this.value);
            document.getElementById('colValue').textContent = colValues[currentColIdx];
            updatePlot();
        });

    </script>
</body>
</html>
"""

# Replace placeholders with actual data
html_content = html_content.replace('DATA_PLACEHOLDER', json.dumps(all_data))
html_content = html_content.replace('TIME_VALUES_PLACEHOLDER', json.dumps([f"{val:.1f}" for val in S_TIME_VALUES]))
html_content = html_content.replace('COL_VALUES_PLACEHOLDER', json.dumps([f"{val:.1f}" for val in S_COL_VALUES]))
html_content = html_content.replace('ROW_VALUES_PLACEHOLDER', json.dumps([f"{val:.1f}" for val in S_ROW_VALUES]))
html_content = html_content.replace('GLOBAL_MIN_PLACEHOLDER', str(global_min))
html_content = html_content.replace('GLOBAL_MAX_PLACEHOLDER', str(global_max))
html_content = html_content.replace('NUM_FRAMES_PLACEHOLDER', str(NUM_FRAMES))

# Write HTML file
plot_path = Path(__file__).parent.resolve() / 'interactive_plot.html'
with plot_path.open(mode='w') as f:
    f.write(html_content)
