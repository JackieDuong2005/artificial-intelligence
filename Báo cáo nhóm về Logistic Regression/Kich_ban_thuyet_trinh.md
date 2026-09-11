# KỊCH BẢN THUYẾT TRÌNH BÀI TẬP LỚN: HỒI QUY LOGISTIC (LOGISTIC REGRESSION)
## ĐỀ TÀI: PHÂN LOẠI ĐỘ CHÍN CỦA QUẢ CAM TRÊN TẬP DỮ LIỆU 1000 QUẢ
### SO SÁNH 3 HÀM LOSS FUNCTION BẰNG THUẬT TOÁN STOCHASTIC GRADIENT DESCENT (SGD)

---
* **Thời lượng dự kiến:** 10 – 12 phút.
* **Phong thái trình bày:** Tự tin, giọng nói mạch lạc, nhấn mạnh vào bản chất toán học và các biểu đồ trực quan.
* **Tài liệu trình chiếu:** File notebook `NhomX.ipynb` chạy trên Visual Studio Code.

---

## ⏱️ PHẦN MỞ ĐẦU: CHÀO HỎI & GIỚI THIỆU NHÓM (1 phút)
*(Trình chiếu: Cell 0 - Tiêu đề & Bảng thông tin nhóm)*

> **🗣️ LỜI THOẠI GỢI Ý:**
> 
> "Kính chào Thầy/Cô và các bạn! 
> 
> Hôm nay, đại diện cho **Nhóm 14**, em xin phép được báo cáo và thuyết minh về đề tài: **'Nghiên cứu mô hình Hồi quy Logistic và so sánh thực nghiệm 3 hàm Loss function khác nhau bằng thuật toán Stochastic Gradient Descent (SGD) tự lập trình từ đầu'**.
> 
> Bài toán thực nghiệm mà nhóm em lựa chọn là bài toán phân loại nhị phân thực tế trong nông nghiệp: **Dự đoán độ chín của Quả Cam (Cam chín vàng / Cam xanh chưa chín)** dựa trên 2 đặc trưng đo đạc tự nhiên là **Khối lượng (Weight)** và **Mã màu sắc vỏ quả (ColorCode)**.
> 
> Toàn bộ mã nguồn của nhóm đã được lập trình hoàn toàn bằng NumPy từ con số 0, không phụ thuộc vào thư viện Scikit-learn, và chạy thành công 100% trên môi trường notebook này. 
> 
> Sau đây, em xin phép đi vào nội dung chi tiết theo từng phần của đề tài."

---

## ⏱️ PHẦN 1: CƠ SỞ LÝ THUYẾT VÀ BẢN CHẤT TOÁN HỌC (3.5 phút)
*(Trình chiếu: Cell 1 - Mục 1.1, 1.2, 1.3)*

### 1.1. Mô hình toán học & Vai trò hàm Sigmoid
*(Cuộn màn hình đến Mục 1.1)*

> **🗣️ LỜI THOẠI GỢI Ý:**
> 
> "Thưa Thầy/Cô, đầu tiên ở Mục 1.1 là cơ sở toán học của mô hình:
> 
> Dữ liệu đầu vào của một quả cam được biểu diễn bằng vector đặc trưng: `x = [x₁, x₂]ᵀ`, trong đó:
> * `x₁` là đại lượng đặc trưng cho **Khối lượng quả cam** tính bằng gram (dao động tự nhiên từ 90g đến 280g).
> * `x₂` là đại lượng đặc trưng cho **Mã màu sắc vỏ quả cam** trên thang đo số từ 0.0 (xanh lục) đến 2.0 (vàng cam chín).
> 
> Mô hình tổ hợp tuyến tính điểm số của quả cam theo công thức:
> **z = a₁·x₁ + a₂·x₂ + b**
> Trong đó `a₁` là trọng số cho khối lượng cam, `a₂` là trọng số cho màu vỏ cam và `b` là hệ số điều chỉnh ngưỡng (bias).
> 
> **[Nhấn mạnh điểm này]:** Con số `z` này có thể nhận bất kỳ giá trị nào từ âm vô cùng đến dương vô cùng, nên không thể diễn giải nó trực tiếp như một xác suất. Do đó, hàm kích hoạt phi tuyến **Sigmoid** xuất hiện như một 'bộ chuyển đổi xác suất':
> **σ(z) = 1 / (1 + e⁻ᶻ)**
> Hàm Sigmoid nén toàn bộ miền giá trị thực về khoảng mở (0, 1). Giá trị đầu ra `p = σ(z)` chính là xác suất dự đoán quả cam đã đạt độ chín.
> 
> Đặc biệt, đạo hàm của hàm Sigmoid có một tính chất giải tích rất đẹp: **σ'(z) = σ(z)·(1 - σ(z))**, tức là lấy chính kết quả vừa tính xong nhân với phần bù của nó, giúp máy tính tiết kiệm tối đa thời gian tính toán."

---

### 1.2. Trình bày và so sánh 3 hàm Loss function
*(Cuộn màn hình đến Mục 1.2 & Bảng Mục 1.3)*

> **🗣️ LỜI THOẠI GỢI Ý:**
> 
> "Ở Mục 1.2, nhóm tiến hành thiết lập giải tích chi tiết cho 3 hàm mất mát theo đúng trình tự nghiên cứu:
> 
> **1. Đầu tiên là hàm Mean Squared Error (MSE) với nhãn 0 và 1:**
> * Công thức Loss: `Loss = (y - p)²` với `p = σ(z)`.
> * Khi áp dụng quy tắc chuỗi để tính đạo hàm theo trọng số, thừa số đạo hàm của Sigmoid là `p·(1 - p)` vẫn còn nguyên trong gradient:
>   `Gradient = -2·(y - p) · p·(1 - p) · x`
> * **[Điểm mấu chốt cần nhấn mạnh]:** Thừa số `p·(1 - p)` này chính là nguyên nhân gây ra **hiện tượng triệt tiêu gradient (Vanishing Gradient)**! Khi mô hình dự đoán rất sai lệch (ví dụ quả cam chín y=1 mà đoán p=0.001), thành phần `p·(1-p)` sẽ xấp xỉ về 0, làm cho gradient gần như biến mất và mô hình bị 'tê liệt', ngừng học dù sai số rất lớn. Ngoài ra, hàm MSE kết hợp Sigmoid là hàm phi lồi (*non-convex*), rất dễ mắc kẹt ở cực tiểu cục bộ.
> 
> **2. Thứ hai là hàm Logistic Loss với nhãn đối xứng {-1, +1}:**
> * Đây là dạng chuẩn tắc xuất phát từ nguyên lý hợp lý cực đại (Maximum Likelihood Estimation).
> * Nhờ dùng nhãn đối xứng `±1`, xác suất cả 2 lớp được gộp chung: `P(y|x) = σ(y·z)`.
> * Hàm Loss cho một quả cam là: `Loss = ln(1 + e⁻ʸᶻ)`.
> * Đạo hàm của nó rất mượt mà: `Gradient = -y·(1 - σ(y·z))·x`. Hàm này lồi toàn cục và không bị triệt tiêu gradient khi dự đoán sai.
> 
> **3. Thứ ba là hàm Binary Cross-Entropy (BCE) với nhãn 0 và 1:**
> * Công thức Loss: `Loss = -[y·ln(p) + (1 - y)·ln(1 - p)]`.
> * Điểm vi diệu nhất của BCE nằm ở bước tính đạo hàm theo quy tắc chuỗi: Mẫu số `p·(1 - p)` sinh ra từ đạo hàm hàm logarit **triệt tiêu hoàn toàn** với tử số `p·(1 - p)` của đạo hàm Sigmoid!
> * Gradient rút gọn còn cực kỳ đơn giản: **Gradient = (p - y)·x**.
> * Lực đẩy gradient lúc này tỉ lệ trực tiếp với độ lệch sai số `(p - y)`. Sai số càng lớn thì mô hình học càng mạnh mẽ, giúp thuật toán hội tụ cực kỳ nhanh và ổn định."

---

## ⏱️ PHẦN 2: TẬP DỮ LIỆU CỦA NHÓM & TIỀN XỬ LÝ (2 phút)
*(Trình chiếu: Cell 3, 4, 5, 6, 7)*

> **🗣️ LỜI THOẠI GỢI Ý:**
> 
> "Tiếp theo ở Phần 2, nhóm em xây dựng bộ dữ liệu riêng gồm **1000 quả cam** mô phỏng phân bố sinh học thực tế:
> * **500 quả Cam chưa chín (cam xanh):** Cân nặng trung bình ~155g, màu vỏ xanh lục với mã màu trung bình 0.45.
> * **500 quả Cam đã chín (cam vàng cam):** Cân nặng trung bình ~205g (do quả chín tích nhiều nước đường ngọt hơn), màu vỏ vàng cam chín với mã màu trung bình 1.55.
> 
> **[Cuộn xuống xem Biểu đồ Scatter 2D & 3D ở Cell 5]:**
> Như Thầy/Cô có thể thấy trên biểu đồ phân bố 2D và 3D, hai nhóm cam phân tách rất rõ ràng: các quả cam xanh phân bố ở góc dưới bên trái, trong khi cam chín tập trung ở góc trên bên phải. Ở giữa có một vùng chồng lấp nhỏ tự nhiên do các quả cam ương, vừa chớm chín. Điều này phản ánh hoàn hảo tính thực tế của nông sản.
> 
> **[Cuộn xuống Cell 6 & 7 - Nhấn mạnh chuẩn hóa Z-score]:**
> Một điểm kỹ thuật cực kỳ quan trọng ở đây là **Chuẩn hóa Z-score**:
> `X_norm = (X - μ) / σ`
> Cân nặng quả cam `x₁` dao động từ 90 đến 280g, trong khi mã màu vỏ `x₂` chỉ từ 0 đến 2. Sự chênh lệch thang đo hàng trăm lần này sẽ làm biến dạng bề mặt hàm Loss thành hình elip rất dẹt, khiến thuật toán Gradient Descent bị dao động rung lắc zíc-zắc và học rất chậm.
> Việc chuẩn hóa Z-score đưa cả 2 đặc trưng của quả cam về cùng thang đo chuẩn (trung bình bằng 0, độ lệch chuẩn bằng 1), giúp bề mặt hàm loss có dạng tròn đối xứng, SGD đi thẳng và hội tụ nhanh hơn gấp nhiều lần."

---

## ⏱️ PHẦN 3: LẬP TRÌNH THUẬT TOÁN TỐI ƯU SGD TỪ ĐẦU (2 phút)
*(Trình chiếu: Cell 8, 9, 10)*

> **🗣️ LỜI THOẠI GỢI Ý:**
> 
> "Ở Phần 3, nhóm hiện thực hóa toàn bộ các công thức giải tích thành code Python bằng thư viện NumPy:
> 
> **[Chỉ vào Cell 9]:**
> * Hàm `sigmoid(z)` được trang bị kỹ thuật `np.clip(z, -250, 250)` để chống tràn số lũy thừa (*overflow warning*) khi z quá lớn hoặc quá nhỏ.
> * Các hàm Loss và Gradient của **MSE**, **Logistic Loss**, và **BCE** được lập trình bám sát 100% công thức đạo hàm đã chứng minh ở Phần 1.
> * Riêng `logistic_loss`, nhóm dùng `np.logaddexp` để tối ưu ổn định số học; hàm `bce_loss` dùng epsilon 10⁻¹⁵ để tránh lỗi logarit của 0.
> 
> **[Chỉ vào Cell 10 - Giải thích hàm train_sgd]:**
> Thuật toán tối ưu mà đề bài yêu cầu là **Stochastic Gradient Descent (SGD)**. 
> Tại mỗi epoch huấn luyện:
> 1. Nhóm dùng `np.random.permutation` để xáo trộn ngẫu nhiên thứ tự 1000 mẫu.
> 2. Duyệt qua từng mẫu đơn lẻ: tính toán sai số, tính vector gradient, và lập tức cập nhật trọng số:
>    `w = w - learning_rate * grad_w`
>    `b = b - learning_rate * grad_b`
> Với 50 epochs, các trọng số được mài giũa và cập nhật liên tục qua đúng **50,000 lần**."

---

## ⏱️ PHẦN 4 & 5: KẾT QUẢ HUẤN LUYỆN VÀ SO SÁNH TRỰC QUAN (3 phút)
*(Trình chiếu: Cell 12, 14, 15, 16)*

> **🗣️ LỜI THOẠI GỢI Ý:**
> 
> "Bây giờ, em xin trình chiếu kết quả huấn luyện thực nghiệm của cả 3 mô hình với cùng siêu tham số: tốc độ học `λ = 0.02` và `50 epochs`:
> 
> **[Chỉ vào Cell 14 - Biểu đồ 1: Loss Curves]:**
> * Đây là biểu đồ so sánh tốc độ hội tụ của 3 hàm Loss qua 50 Epochs.
> * Đường cong của **BCE** (nét đứt xanh lá) và **Logistic Loss** (xanh dương) giảm rất mượt mà, lao dốc dứt khoát ngay từ những epoch đầu tiên và nhanh chóng đạt trạng thái hội tụ ổn định.
> * Trong khi đó, **MSE** (đường chấm đỏ) có tốc độ suy giảm chậm hơn ở các bước đầu do ảnh hưởng của hiện tượng triệt tiêu gradient.
> 
> **[Chỉ vào Cell 15 - Biểu đồ 2: Decision Boundaries]:**
> * Đây là đường ranh giới phân loại của 3 mô hình trên không gian 1000 điểm dữ liệu chuẩn hóa.
> * Cả 3 mô hình đều tìm được đường ranh giới thẳng băng chia tách hai đám mây điểm đỏ (chưa chín) và xanh (đã chín) cực kỳ chuẩn xác.
> * Đáng chú ý, trọng số sau huấn luyện của `ColorCode` (w₂ ≈ 5.40) lớn hơn nhiều so với `Weight` (w₁ ≈ 2.52). Điều này chứng minh một cách định lượng rằng: **Màu sắc chính là đặc trưng mang tính quyết định mạnh hơn cân nặng** trong việc phân biệt độ chín.
> 
> **[Chỉ vào Cell 16 - Bảng đánh giá chỉ số & Ma trận nhầm lẫn]:**
> * Cả 3 mô hình đều đạt hiệu năng xuất sắc:
>   * **Độ chính xác (Accuracy): 96.20%** trên toàn bộ 1000 quả cam.
>   * **Precision: 96.39%**, **Recall: 96.00%**, và **F1-Score: 96.19%**.
> * Ma trận nhầm lẫn cho thấy trong 1000 quả cam, mô hình chỉ phân loại nhầm khoảng 38 quả nằm sát vùng ranh giới do hiện tượng chồng lấp tự nhiên của nông sản.
> * Đặc biệt, khi đem thử nghiệm trên **50 quả cam mới toanh chưa từng xuất hiện trong tập huấn luyện (Test Set)** ở Cell 12, mô hình đạt độ chính xác lên tới **98.00%** (đoán đúng 49/50 quả cam), khẳng định mô hình có độ khái quát hóa tuyệt vời."

---

## ⏱️ PHẦN 6: KẾT LUẬN & TRẢ LỜI CÂU HỎI VẤN ĐÁP (1.5 phút)
*(Trình chiếu: Cell 17)*

> **🗣️ LỜI THOẠI GỢI Ý:**
> 
> "Tóm lại, qua bài tập lớn này, nhóm em rút ra 3 kết luận học thuật quan trọng:
> 
> 1. **Binary Cross-Entropy (BCE) và Logistic Loss là sự lựa chọn tối ưu nhất** cho bài toán phân loại nhị phân vì đảm bảo tính lồi toàn cục và lực đẩy gradient luôn tỉ lệ trực tiếp với sai số, giúp mô hình học nhanh và ổn định.
> 2. **MSE không phù hợp cho bài toán phân loại kết hợp Sigmoid** do bề mặt hàm loss phi lồi và mắc phải hiện tượng triệt tiêu gradient khi mô hình dự đoán rất sai lệch.
> 3. Thuật toán **SGD kết hợp chuẩn hóa Z-score** tự xây dựng từ đầu hoạt động cực kỳ hiệu quả, hội tụ ổn định và phân loại chính xác bài toán thực tế.
> 
> Phần thuyết minh của Nhóm 14 đến đây là kết thúc. Em xin chân thành cảm ơn Thầy/Cô và các bạn đã lắng nghe. Nhóm em rất mong nhận được câu hỏi và nhận xét từ Thầy/Cô!"

---

## 💡 "PHAO CỨU SINH": BỘ 5 CÂU HỎI VẤN ĐÁP THẦY CÔ HAY HỎI NHẤT & CÁCH TRẢ LỜI ĂN ĐIỂM TỐI ĐA

### ❓ Câu 1: Tại sao phải dùng hàm Sigmoid mà không dùng trực tiếp hàm tuyến tính z?
> **👉 Trả lời ngắn gọn:** Hàm tuyến tính `z = a₁·x₁ + a₂·x₂ + b` có miền giá trị từ `-∞` đến `+∞`, không thể biểu diễn xác suất. Hàm Sigmoid ánh xạ toàn bộ miền giá trị thực về khoảng `(0, 1)`, giúp ta diễn giải đầu ra thành xác suất quả chín từ 0% đến 100%, từ đó đưa ra quyết định phân loại rất tự nhiên với ngưỡng 0.5.

### ❓ Câu 2: Chứng minh tại sao gradient của BCE lại không còn đạo hàm của hàm Sigmoid?
> **👉 Trả lời ngắn gọn:** Theo quy tắc chuỗi: `dLoss/dz = (dLoss/dp) · (dp/dz)`. Đạo hàm của BCE theo p là `(p - y) / [p(1 - p)]`. Đạo hàm của Sigmoid theo z là `p(1 - p)`. Khi nhân lại với nhau, thừa số `p(1 - p)` ở mẫu số triệt tiêu hoàn toàn với tử số, chỉ còn lại `(p - y)`. Do đó gradient không còn đạo hàm Sigmoid.

### ❓ Câu 3: Hiện tượng triệt tiêu Gradient (Vanishing Gradient) ở MSE xảy ra khi nào?
> **👉 Trả lời ngắn gọn:** Trong MSE, gradient có dạng `-2·(y - p)·p(1 - p)·x`. Khi mô hình dự đoán rất sai (ví dụ nhãn thật y=1 nhưng mô hình đoán p=0.001), thành phần `p(1 - p)` xấp xỉ bằng 0. Khi nhân vào, nó triệt tiêu gradient về gần bằng 0, khiến các trọng số hầu như không được cập nhật dù sai số thực tế rất lớn.

### ❓ Câu 4: Sự khác nhau cơ bản giữa Batch Gradient Descent, Mini-batch GD và Stochastic GD (SGD)?
> **👉 Trả lời ngắn gọn:** 
> * **Batch GD:** Tính đạo hàm trên toàn bộ 1000 mẫu rồi mới cập nhật trọng số 1 lần (chậm chạp).
> * **SGD (nhóm dùng):** Cứ xem từng mẫu đơn lẻ là cập nhật trọng số ngay (nhanh, có tính ngẫu nhiên giúp thoát cực tiểu cục bộ).
> * **Mini-batch GD:** Cập nhật theo từng cụm nhỏ (ví dụ 16, 32 mẫu) để dung hòa ưu điểm của cả hai.

### ❓ Câu 5: Tại sao nhóm không chia dữ liệu 70-15-15 Train-Val-Test mà lại train trên 1000 mẫu?
> **👉 Trả lời ngắn gọn:** Mô hình Logistic Regression với 2 đặc trưng chỉ có 3 tham số, ranh giới phân lớp là một đường thẳng nên gần như không thể bị Overfitting (học vẹt). Trọng tâm đề bài là so sánh hành vi của 3 hàm loss trên cùng tập dữ liệu. Ngoài ra, nhóm đã tạo thêm một tập Test riêng gồm 50 quả độc lập và đạt độ chính xác 98%, chứng minh mô hình khái quát hóa rất tốt mà không cần chia tập validation.
