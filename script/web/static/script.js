document.addEventListener('DOMContentLoaded', function () {
    // 電池健康度滑桿顯示
    const batteryHealthSlider = document.getElementById('battery-health');
    const batteryHealthValue = document.getElementById('battery-health-value');

    batteryHealthSlider.addEventListener('input', function () {
        batteryHealthValue.textContent = this.value + '%';
    });

    // 設置日期限制
    const dateInput = document.getElementById('date');

    // 設定今天的日期為預設值
    const today = new Date();
    const formattedDate = today.toISOString().split('T')[0];

    dateInput.value = formattedDate;
    dateInput.min = formattedDate;
    dateInput.max = "2030-12-31";

    // 確保日期在允許範圍內
    if (formattedDate >= dateInput.min && formattedDate <= dateInput.max) {
        dateInput.value = formattedDate;
    } else {
        // 如果今天的日期超出範圍，設為允許範圍的最大值
        dateInput.value = dateInput.max;
    }





    batteryHealthSlider.addEventListener('input', function () {
        batteryHealthValue.textContent = this.value + '%';
    });

    // 廠商選擇更新型號
    const vendorSelect = document.getElementById('vendor');
    const modelSelect = document.getElementById('model');

    vendorSelect.addEventListener('change', function () {

        const vendor = this.value;
        fetch(`/get_models?vendor=${encodeURIComponent(vendor)}`)
            .then(response => response.json())
            .then(models => {
                modelSelect.innerHTML = '';
                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                });
                // 觸發型號變更事件以更新容量
                modelSelect.dispatchEvent(new Event('change'));
            })
            .catch(error => {
                console.error('Error fetching models:', error);
            });
    });

    // 型號選擇更新容量
    const capacitySelect = document.getElementById('capacity');

    modelSelect.addEventListener('change', function () {
        const vendor = vendorSelect.value;
        const model = this.value;
        // 使用 encodeURIComponent 對參數進行編碼
        fetch(`/get_capacities?vendor=${encodeURIComponent(vendor)}&model=${encodeURIComponent(model)}`)
            .then(response => response.json())
            .then(capacities => {
                capacitySelect.innerHTML = '';
                capacities.forEach(capacity => {
                    const option = document.createElement('option');
                    option.value = capacity;
                    option.textContent = capacity;
                    capacitySelect.appendChild(option);
                });
            })
            .catch(error => {
                console.error('Error fetching capacities:', error);
            });
    });



    // 表單提交處理
    const form = document.getElementById('prediction-form');
    const predictionResult = document.getElementById('prediction-result');
    const predictedPrice = document.getElementById('predicted-price');
    const productInfo = document.getElementById('product-info');
    const batteryInfo = document.getElementById('battery-info');
    const predictionDate = document.getElementById('prediction-date');
    const submitButton = form.querySelector('button[type="submit"]');

    // 新增變數來追蹤上次點擊時間
    let lastClickTime = 0;

    form.addEventListener('submit', function (e) {
        e.preventDefault();

        // 檢查是否在一秒內重複點擊
        const currentTime = new Date().getTime();
        if (currentTime - lastClickTime < 1000) {
            // 使用 window.alert 確保警告訊息顯示
            window.alert('請稍候再試，點擊過於頻繁！');
            return false; // 確保事件不繼續傳播
        }

        // 更新上次點擊時間
        lastClickTime = currentTime;

        // 禁用按鈕，防止重複提交
        submitButton.disabled = true;
        submitButton.textContent = '處理中...';

        const formData = new FormData(this);

        fetch('/', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                // 檢查回應狀態
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // 檢查回應中是否有錯誤訊息
                if (data.error) {
                    throw new Error(data.error);
                }

                // 顯示預測結果
                predictionResult.style.display = 'block';
                predictedPrice.textContent = data.預測金額.toLocaleString();

                // 修改這部分 - 處理小數點容量的顯示
                const capacityDisplay = parseFloat(data.容量) < 1 ?
                    `${(data.容量 * 1024).toFixed(0)}MB` :
                    `${data.容量}GB`;

                productInfo.textContent = `${data.廠商} ${data.型號} ${capacityDisplay}`;
                batteryInfo.textContent = `${data.電池健康度}%`;

                // 格式化日期
                const date = new Date(data.日期);
                const formattedDate = `${date.getFullYear()}年${date.getMonth() + 1}月${date.getDate()}日`;
                predictionDate.textContent = formattedDate;

                // 滾動到結果區域
                predictionResult.scrollIntoView({ behavior: 'smooth' });
            })
            .catch(error => {
                console.error('Error:', error);
                if (error.message !== 'undefined') {
                    window.alert('預測過程中發生錯誤，請稍後再試。');
                }
            })
            .finally(() => {
                // 一秒後重新啟用按鈕
                setTimeout(() => {
                    submitButton.disabled = false;
                    submitButton.textContent = '預測';
                }, 500);
            });
    });



    // 初始加載時觸發廠商變更事件
    vendorSelect.dispatchEvent(new Event('change'));
});
