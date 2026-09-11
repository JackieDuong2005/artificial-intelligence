# BẢNG TỔNG HỢP CÁC THÔNG SỐ & SỐ LIỆU CỐT LÕI CẦN NHỚ (CHEAT SHEET)
## ĐỀ TÀI: HỒI QUY LOGISTIC (LOGISTIC REGRESSION) - PHÂN LOẠI ĐỘ CHÍN QUẢ CAM
### NHÓM 14 - BÀI TẬP LỚN MÔN TRÍ TUỆ NHÂN TẠO

---

## 🍊 1. BỘ DỮ LIỆU THỰC TẾ (1000 QUẢ CAM)
* **Quy mô tập dữ liệu:** **1000 quả cam**, chia đều thành 2 lớp:
  * **500 quả Cam chưa chín (Class 0 / Nhãn 0 hoặc -1):**
    * Cân nặng trung bình: **155g** (độ lệch chuẩn 28g).
    * Mã màu sắc vỏ quả trung bình: **0.45** (vỏ màu xanh lục).
  * **500 quả Cam đã chín (Class 1 / Nhãn 1 hoặc +1):**
    * Cân nặng trung bình: **205g** (cam mọng nước ngọt hơn nên nặng hơn).
    * Mã màu sắc vỏ quả trung bình: **1.55** (vỏ màu vàng cam chín).
* **2 đặc trưng đo đạc đầu vào (x = [x₁, x₂]ᵀ):**
  * `x₁` (Khối lượng - Weight): Dao động tự nhiên từ **90g đến 280g**.
  * `x₂` (Mã màu sắc vỏ - ColorCode): Thang đo liên tục từ **0.0 (xanh lục) đến 2.0 (vàng cam chín)**.
* **Độ nhiễu thực tế:** Có hiện tượng chồng lấn tự nhiên giữa 2 nhóm (cam chín vỏ còn xanh, cam rám nắng vàng sớm, cam ương) giúp mô hình chống học vẹt (Overfitting).

---

## 📊 2. TIỀN XỬ LÝ DỮ LIỆU (CHUẨN HÓA Z-SCORE)
* **Công thức:** `X_norm = (X - μ) / σ`
* **Lý do bắt buộc phải chuẩn hóa:** Cân nặng (90-280g) lớn gấp hàng trăm lần Mã màu vỏ (0-2). Nếu không chuẩn hóa, gradient theo cân nặng sẽ quá lớn làm thuật toán dao động giật cục, còn màu sắc học rất chậm.
* **Kết quả sau chuẩn hóa:** Đưa cả 2 đặc trưng về thang đo chuẩn có **trung bình bằng 0** và **độ lệch chuẩn bằng 1**, giá trị nằm gọn trong khoảng **-2.5 đến +2.5**.

---

## ⚙️ 3. BỘ SIÊU THAM SỐ HUẤN LUYỆN (HYPERPARAMETERS)
* **Tốc độ học (Learning Rate):** **λ = 0.02** (`LR = 0.02`).
  * *Lý do chọn 0.02:* Đây là điểm cân bằng hoàn hảo (sweet spot). Nếu chọn lớn (0.5, 1.0) sẽ bị nhảy vọt qua đáy (overshoot) và phân kỳ; nếu chọn quá nhỏ (0.0001) sẽ học quá chậm như kiến bò.
* **Số lượt lặp huấn luyện:** **50 Epochs**.
* **Tổng số lần cập nhật trọng số SGD:** **50,000 lần** (`50 epochs × 1000 quả cam` cập nhật liên tục sau từng quả).
* **Khởi tạo trọng số ban đầu:** Rất nhỏ quanh số 0: **w ~ N(0, 0.01²)** và **bias b = 0.0**.
  * *Lý do:* Tránh rơi vào vùng bão hòa của hàm Sigmoid làm triệt tiêu đạo hàm ngay từ đầu.
* **Hạt giống ngẫu nhiên cố định:** **random_seed = 42** (đảm bảo tính tái lập kết quả 100% trên mọi máy tính).

---

## 🎯 4. KẾT QUẢ HIỆU NĂNG & ĐỘ CHÍNH XÁC (METRICS)
* **Trên tập huấn luyện (1000 quả cam):**
  * **Độ chính xác (Train Accuracy):** **96.20%** cho cả 3 mô hình (đoán đúng 962 quả, nhầm 38 quả nằm ở vùng ranh giới nhiễu).
  * **Precision:** **96.39%**
  * **Recall:** **96.00%**
  * **F1-Score:** **96.19%**
* **Trên tập kiểm thử độc lập (Test Set - 50 quả cam mới toanh):**
  * **Số lượng:** **50 quả** (25 cam xanh, 25 cam chín, hạt giống seed = 2026).
  * **Độ chính xác (Test Accuracy):** **98.00%** (đoán đúng **49 / 50 quả** cam mới).
* **Ngưỡng quyết định phân loại:** Xác suất **0.5 (50%)** (Nếu p >= 0.5 ➔ Cam chín; p < 0.5 ➔ Cam xanh).

---

## ⚖️ 5. BỘ TRỌNG SỐ TỐI ƯU w = [a₁, a₂] VÀ Ý NGHĨA THỰC TẾ
* **Hàm 2 (Logistic Loss) & Hàm 3 (Binary Cross-Entropy):**
  * Vector trọng số: **w = [a₁ = 2.52, a₂ = 5.40]**, Bias: **b = 0.08**
  * Loss giảm từ **0.2192 ➔ 0.0861**
* **Hàm 1 (Mean Squared Error - MSE):**
  * Vector trọng số: **w = [a₁ = 2.04, a₂ = 3.98]**, Bias: **b = 0.03**
  * Loss giảm từ **0.0839 ➔ 0.0260**
* 💡 **Ý nghĩa thực tế đắt giá nhất:**
  Trọng số của màu sắc vỏ cam **a₂ ≈ 5.4 lớn hơn gấp đôi** trọng số khối lượng **a₁ ≈ 2.5**. Điều này chứng minh định lượng rằng: **Màu sắc vỏ quả là đặc trưng mang tính quyết định độ chín mạnh hơn nhiều so với cân nặng!**

---

## 📈 6. CÁC MỐC TRỰC QUAN QUAN TRỌNG TRÊN BIỂU ĐỒ
1. **Biểu đồ đường cong mất mát (Loss Curves):**
   * Đường xanh lá (BCE) và xanh dương (Logistic Loss) **trùng khít 100% lên nhau** do cùng xuất phát từ nguyên lý hợp lý cực đại (MLE).
   * Cả 2 đường đều **lao dốc đứng ở 5 epoch đầu tiên** (từ 0.22 rơi xuống dưới 0.10) vì gradient tỉ lệ trực tiếp với sai số (p - y).
   * Cả 3 đường đều **chạm đáy phẳng lì từ epoch 35 đến 50** (chứng minh mô hình đã hội tụ tối ưu).
2. **Biểu đồ ranh giới phân loại (Decision Boundary):**
   * Phương trình đường ranh giới tại ngưỡng xác suất 50%: `z = a₁·x₁ + a₂·x₂ + b = 0 ➔ x₂ = - (a₁/a₂)·x₁ - (b/a₂)`.
   * Đường phân chia của Logistic Loss và BCE trùng khớp hoàn toàn.
3. **Ma trận nhầm lẫn (Confusion Matrix):**
   * Phân loại chính xác 962 quả, chỉ nhầm 38 quả do yếu tố sinh học tự nhiên.

---

## 🛡️ 7. KỸ THUẬT LẬP TRÌNH AN TOÀN TRONG CODE
* **Hàm Sigmoid:** Sử dụng `np.clip(z, -250.0, 250.0)` để khóa miền giá trị, chống lỗi tràn số lũy thừa máy tính (`overflow warning`) trong `np.exp`.
* **Hàm BCE:** Sử dụng `epsilon = 1e-15` để kẹp an toàn cho xác suất p, tránh lỗi `ln(0) = -inf`.
* **Hàm Logistic Loss:** Sử dụng `np.logaddexp(0.0, -y*z)` để tối ưu hóa tính toán số học ổn định.

---

## 🧠 8. ĐOẠN TÓM TẮT 30 GIÂY NHẨM TRONG ĐẦU TRƯỚC KHI THUYẾT TRÌNH
> *"Bài toán phân loại **1000 quả cam** (500 cam xanh 155g, 500 cam chín 205g) dựa trên Cân nặng và Màu sắc vỏ. Chuẩn hóa Z-score để đưa về cùng thang đo. Huấn luyện bằng thuật toán SGD với **LR = 0.02** qua **50 epochs** (50,000 lần cập nhật). Kết quả đạt **96.20% train**, **98.00% test (49/50 quả)**. Màu sắc quan trọng gấp đôi cân nặng (trọng số **5.4 so với 2.5**). BCE và Logistic Loss trùng khớp hoàn toàn trên biểu đồ thực nghiệm!"*
