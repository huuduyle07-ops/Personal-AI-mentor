# Personal Accountability AI

[English](RESEARCH_ROADMAP.md) | [한국어](RESEARCH_ROADMAP.ko.md) | [Tiếng Việt](RESEARCH_ROADMAP.vi.md)

![Python Versions](https://img.shields.io/badge/Python-3.11-blue)

**Trợ lý AI cá nhân có khả năng thích ứng, ưu tiên xử lý cục bộ, theo dõi tiến độ, đưa ra quyết định dựa trên hành vi và hỗ trợ học tập liên tục.**

**Phiên bản:** 0.2 — Bản thảo kiến trúc  
**Trạng thái:** Nghiên cứu và lập kế hoạch  
**Thời gian thử nghiệm ban đầu:** 7 ngày  
**Chiến lược phát triển:** Python Prototype → Nghiên cứu mô hình → Rust Runtime

---

## 1. Tổng quan dự án

Personal Accountability AI là một trợ lý chủ động được thiết kế nhằm theo dõi tiến độ hằng ngày, xác minh kết quả người dùng báo cáo, nhận diện các thay đổi trong hành vi và điều chỉnh cách tương tác theo hoàn cảnh thực tế.

Khác với những chatbot nhắc việc thông thường, hệ thống hướng đến việc kết hợp:

- **Accountability — Trách nhiệm với mục tiêu:** Theo dõi những cam kết và mục tiêu hằng ngày.
- **Adaptation — Khả năng thích ứng:** Điều chỉnh tương tác dựa trên hành vi và hoàn cảnh của người dùng.
- **Verification — Xác minh:** Đánh giá mức độ hoàn thành nhiệm vụ và kết quả học tập.
- **Persistent Memory — Bộ nhớ lâu dài:** Duy trì lịch sử tương tác, tiến độ và trạng thái kiến thức.
- **Efficient Intelligence — Suy luận tiết kiệm:** Hạn chế những lần gọi mô hình ngôn ngữ lớn (LLM) không cần thiết.
- **Continuous Experimentation — Thử nghiệm liên tục:** Cải thiện thiết kế dựa trên những kết quả có thể đo lường.

Phiên bản đầu tiên được phát triển để phục vụ thử nghiệm cá nhân, không nhằm mục đích thương mại hoặc giám sát người khác.

### Triết lý cốt lõi

> Quan sát → Thấu hiểu → Quyết định → Xác minh → Thích ứng → Đánh giá

Một thử nghiệm thất bại không đồng nghĩa với việc cả dự án thất bại.

Mỗi thử nghiệm cần tạo ra bằng chứng, phát hiện giới hạn hoặc đặt ra những câu hỏi nghiên cứu mới.

## 2. Mục tiêu dự án

Hệ thống hướng đến việc nghiên cứu liệu một trợ lý có khả năng thích ứng có thể:

1. Chủ động yêu cầu người dùng báo cáo tiến độ hằng ngày.
2. Phân biệt việc hoàn thành mục tiêu với những biến động bất thường trong hành vi.
3. Đánh giá thông tin tự báo cáo mà không mặc định người dùng thiếu trung thực.
4. Nhận diện những dấu hiệu cho thấy người dùng đang quên kiến thức đã học.
5. Điều chỉnh lịch trình theo hoàn cảnh như thiếu ngủ, làm thêm muộn hoặc có công việc đột xuất.
6. Giảm chi phí suy luận thông qua bộ định tuyến quyết định thích ứng.
7. Chuyển những tình huống phức tạp hoặc không rõ ràng sang LLM phù hợp.
8. Học từ những lần tương tác trước mà không làm mất quyền tự chủ của người dùng.

## 3. Kiến trúc hệ thống dự kiến

```text
Discord / iOS Shortcuts
          |
          v
    Event Ingestion
      (Nhận sự kiện)
          |
          v
    Priority Queue
  GENERAL / SYSTEM / URGENT
          |
          v
    Feature Extractor
     (Trích đặc trưng)
          |
          v
    Decision Router
          |
    +-----+-------------+
    |                   |
    v                   v
Rule Engine       Decision Model
    |              (Tiny Mamba -
    |                thử nghiệm)
    |                   |
    |             Action Validator
    |                   |
    +---------+---------+
              |
              v
       Action Executor
              |
    +---------+-----------+
    |         |           |
    v         v           v
Templates  Local LLM  Deferred Queue
              |
        Optional Cloud LLM
              |
              v
       SQLite Memory
              |
              v
      Feedback & Metrics
```

Kiến trúc tách biệt các hoạt động mang tính xác định của hệ thống khỏi những quyết định sử dụng mô hình Machine Learning.

**Yêu cầu cốt lõi:** Các quy tắc an toàn, kiểm tra quyền hạn và giới hạn thực thi phải hoạt động độc lập với chính sách quyết định do mô hình học được.

### 3.1. Hệ thống ưu tiên sự kiện

Ứng dụng sử dụng ba cấp độ ưu tiên logic, lấy cảm hứng từ cơ chế lập lịch của hệ điều hành.

| Mức ưu tiên | Phân loại | Ví dụ |
|---|---|---|
| P1 | GENERAL | Báo cáo hằng ngày, câu hỏi thông thường |
| P2 | SYSTEM | Xác minh, thử lại tác vụ, ôn tập theo lịch |
| P3 | URGENT | Các sự kiện cần được xử lý ngay |

Mức ưu tiên của sự kiện và quyết định gọi LLM là hai vấn đề độc lập.

Các sự kiện URGENT phải có cơ chế xử lý dự phòng mang tính xác định khi dịch vụ suy luận không khả dụng.

Scheduler cần có cơ chế đảm bảo tính công bằng, tránh tình trạng các sự kiện ưu tiên thấp bị trì hoãn vô thời hạn (*starvation*).

Hệ thống ưu tiên được triển khai ở tầng ứng dụng và không yêu cầu FreeRTOS.

## 4. Mô hình ra quyết định thích ứng

### 4.1. Tại sao lựa chọn Mamba?

Mamba là một kiến trúc Selective State Space Model (SSM), được thiết kế để xử lý và mô hình hóa dữ liệu tuần tự.

Dự án nghiên cứu khả năng sử dụng một mô hình Mamba nhỏ làm **chính sách ra quyết định theo chuỗi (Sequential Decision Policy)** thay vì dùng nó như một chatbot hội thoại đa năng.

Mô hình được kỳ vọng học những mối quan hệ giữa:

- Lịch sử hành vi của người dùng.
- Các sự kiện và hoạt động được báo cáo.
- Xu hướng hoàn thành nhiệm vụ.
- Đặc điểm của tin nhắn.
- Những hành động trước đây của trợ lý.
- Kết quả của các quyết định trước đó.

Đầu ra của mô hình chỉ là một hành động được đề xuất, không phải một lệnh thực thi trực tiếp.

### 4.2. Không gian hành động

Không gian hành động ban đầu gồm:

```python
ACTIONS = [
    "RULE_RESPONSE",
    "ASK_FOLLOWUP",
    "LOCAL_LLM",
    "DEFER",
    "REQUEST_REVIEW"
]
```

Ý nghĩa:

| Hành động | Mô tả |
|---|---|
| `RULE_RESPONSE` | Sử dụng quy tắc hoặc câu trả lời có sẵn |
| `ASK_FOLLOWUP` | Đặt thêm câu hỏi xác minh |
| `LOCAL_LLM` | Chuyển sang LLM chạy cục bộ |
| `DEFER` | Trì hoãn xử lý đến thời điểm khác |
| `REQUEST_REVIEW` | Yêu cầu xem xét bổ sung hoặc phê duyệt |

Việc sử dụng Cloud LLM được kiểm soát bởi chính sách quyền hạn và an toàn riêng.

Decision Model không được tự ý thực thi lệnh, mở rộng không gian hành động hoặc vượt qua những giới hạn hệ thống.

### 4.3. Biểu diễn dữ liệu đầu vào

Các thử nghiệm ban đầu sẽ sử dụng những đặc trưng hành vi có cấu trúc, thay vì trực tiếp đưa toàn bộ câu chữ vào Mamba.

Những đặc trưng dự kiến:

```text
message_length
message_type
task_completion_ratio
sleep_duration
energy_level
response_delay
recent_completion_trend
previous_followup_count
verification_status
time_since_last_checkin
```

Các giá trị bị thiếu phải được biểu diễn rõ ràng.

Thông tin do người dùng tự báo cáo, kết luận do mô hình suy ra và thông tin đã xác minh phải được phân biệt với nhau.

### 4.4. Đường cơ sở hành vi (Behavioral Baseline)

Hệ thống duy trì những đường cơ sở từ lịch sử hoạt động tương đương.

Ví dụ, với mục tiêu học 30 phút mỗi ngày:

| Thời gian báo cáo | Diễn giải |
|---|---|
| 5 phút | Chưa đạt mục tiêu |
| 30 phút | Hoàn thành mục tiêu |
| 90 phút | Hoàn thành, có biến động tích cực |

Một kết quả cao bất thường không được tự động làm tăng mục tiêu của những ngày tiếp theo.

Hành vi bất thường chỉ là một quan sát cần được giải thích, không phải bằng chứng của sự thiếu trung thực.

Trong giai đoạn *cold start*, khi chưa có đủ dữ liệu lịch sử, hệ thống phải ghi nhận mức độ không chắc chắn thay vì đưa ra kết luận tự tin.

## 5. Chiến lược huấn luyện mô hình

### Giai đoạn A — Supervised Baselines

Xây dựng một tập dữ liệu chứa các chuỗi sự kiện hành vi cùng nhãn quyết định tương ứng.

Các phương pháp cần so sánh:

- Rule-based Routing.
- Logistic Regression.
- Decision Tree.
- Small MLP.
- GRU.
- Tiny Mamba.

Dữ liệu mô phỏng có thể được sử dụng để kiểm tra thuật toán và quá trình triển khai.

Tuy nhiên, kết quả trên dữ liệu mô phỏng không được coi là bằng chứng về hiệu quả thực tế với người dùng.

### Giai đoạn B — On-Policy Distillation (OPD)

Nghiên cứu phương pháp huấn luyện Mamba Student bằng hướng dẫn từ một Teacher Policy mạnh hơn.

Quy trình huấn luyện dự kiến:

1. Student quan sát trạng thái của môi trường mô phỏng.
2. Student lựa chọn một hành động.
3. Môi trường tạo ra trạng thái tiếp theo.
4. Teacher đánh giá trạng thái mà Student gặp phải và cung cấp phân phối hành động mục tiêu.
5. Student cập nhật chính sách bằng một hàm mất mát distillation phù hợp.
6. Lặp lại quá trình và đánh giá trên những tình huống chưa xuất hiện trong quá trình huấn luyện.

Một hàm mục tiêu có thể thử nghiệm:

\[
\mathcal{L}_{OPD}
=
\mathbb{E}_{s\sim d^{\pi_\theta}}
\left[
D_{KL}
\left(
\pi_T(\cdot \mid s)
\parallel
\pi_\theta(\cdot \mid s)
\right)
\right]
\]

Trong đó:

- \(\pi_\theta\): Chính sách của Student.
- \(\pi_T\): Chính sách của Teacher.
- \(d^{\pi_\theta}\): Phân phối trạng thái sinh ra từ các lượt tương tác của Student.

Thành phần *on-policy* yêu cầu thu thập những trạng thái xuất hiện khi chính sách hiện tại của Student đang hoạt động.

Nếu Teacher chỉ cung cấp nhãn hành động rời rạc, thử nghiệm cần được mô tả chính xác là *On-Policy Imitation Learning*, thay vì Distributional Policy Distillation.

Đánh giá của Teacher không mặc nhiên là ground truth.

### Giai đoạn C — Đánh giá Critic tùy chọn

Nghiên cứu liệu một Critic độc lập có cải thiện chất lượng quyết định hay không.

Các vai trò dự kiến:

- **Mamba Actor:** Đề xuất hành động.
- **MLP hoặc LSTM Critic:** Ước lượng chất lượng hành động.
- **Deterministic Validator:** Kiểm tra và áp dụng những giới hạn hệ thống.

Một hàm đánh giá hành động có thể được biểu diễn:

\[
Q_\phi(s_t,a_t)
\]

Critic phải được huấn luyện từ phản hồi có ý nghĩa, kết quả quan sát thực tế hoặc nhãn mục tiêu đã được kiểm chứng.

Ba cấu hình cần đánh giá:

| Thử nghiệm | Kiến trúc |
|---|---|
| A | Mamba + Rule Validator |
| B | Mamba + MLP Critic |
| C | Mamba + LSTM Critic |

LSTM Critic chỉ là thành phần thử nghiệm, không phải dependency bắt buộc.

Chỉ nên giữ lại LSTM nếu những cải thiện đo lường được đủ để bù đắp chi phí tính toán và độ phức tạp bổ sung.

## 6. Quy trình theo dõi tiến độ hằng ngày

### 6.1. Kiểm tra giữa ngày

Thời điểm mặc định: **12:00 trưa**, có thể tùy chỉnh.

Trợ lý yêu cầu báo cáo tiến độ và có thể:

- Cập nhật trạng thái hoàn thành nhiệm vụ.
- Xác minh kết quả được báo cáo.
- Phát hiện trở ngại.
- Ôn lại kiến thức đã học trước đó.
- Đề xuất điều chỉnh cho thời gian còn lại trong ngày.

### 6.2. Tổng kết buổi tối

Trợ lý tổng kết:

- Những mục tiêu đã hoàn thành.
- Những nhiệm vụ chưa hoàn thành.
- Các trở ngại được xác định.
- Kiến thức cần được ôn tập.
- Những điều chỉnh tiềm năng cho ngày hôm sau.

Các hoạt động kiểm tra liên quan nên được gộp trong cùng một phiên tương tác khi có thể, nhằm giảm số lượng thông báo và những lần suy luận không cần thiết.

### 6.3. Recovery Mode

Trợ lý có thể điều chỉnh lịch trình nếu người dùng báo cáo:

- Thiếu ngủ.
- Làm việc đến khuya.
- Bị ốm.
- Có trách nhiệm đột xuất.
- Mức năng lượng thấp bất thường.

Những thay đổi được đề xuất có thể gồm:

- Giảm khối lượng công việc.
- Dời thời điểm check-in.
- Tăng thời gian nghỉ ngơi.
- Chuyển sang mục tiêu nhỏ hơn.

Hệ thống không nên đánh thức người dùng chỉ để kiểm tra xem họ có đang ngủ hay không.

## 7. Xác minh tiến độ và bộ nhớ học tập

Hệ thống phân biệt hoạt động do người dùng báo cáo với kết quả học tập đã được xác minh.

Các trạng thái xác minh ban đầu:

| Trạng thái | Ý nghĩa |
|---|---|
| `reported` | Hoạt động do người dùng tự báo cáo |
| `verified` | Đã vượt qua bài kiểm tra xác minh |
| `passed_after_support` | Đạt sau khi được hướng dẫn bổ sung |
| `review_required` | Cần luyện tập hoặc kiểm tra lại |
| `unknown` | Chưa có đủ bằng chứng |

Lịch sử xác minh phải được bảo tồn.

Một kiến thức từng được xác minh có thể trở lại trạng thái `review_required` nếu những quan sát sau này cho thấy người dùng đã quên.

Các Teacher Harness trong tương lai có thể truy cập phần bộ nhớ chung liên quan và trả về kết quả học tập có cấu trúc.

Những module chuyên môn không được tự ý ghi đè dữ liệu không liên quan.

## 8. Deferred LLM Evaluation

Khi Local LLM không khả dụng hoặc một sự kiện không yêu cầu xử lý ngay, hệ thống có thể đưa tác vụ vào hàng đợi để đánh giá sau.

Ví dụ:

```json
{
  "event_type": "behavior_review",
  "priority": "SYSTEM",
  "status": "pending",
  "retry_slot": "evening",
  "attempts": 0
}
```

Các tác vụ bị trì hoãn cần có:

- Giới hạn số lần thử lại.
- Phát hiện và tránh sự kiện trùng lặp.
- Chính sách hết hạn.
- Ghi log lỗi.
- Duy trì ngữ cảnh liên quan.
- Hủy tác vụ khi sự kiện không còn phù hợp.

Những phản hồi quan trọng liên quan đến an toàn không được phụ thuộc hoàn toàn vào suy luận trì hoãn.

## 9. Tích hợp iOS Shortcuts

Các automation tùy chọn trong Shortcuts có thể gửi sự kiện khi người dùng mở hoặc đóng một ứng dụng.

Ví dụ ban đầu gồm TikTok và YouTube, tùy lựa chọn của người dùng.

Hệ thống phải:

- Yêu cầu người dùng tự cấu hình và cấp quyền.
- Chỉ thu thập dữ liệu cần thiết.
- Xác thực các sự kiện được gửi đến.
- Cho phép tạm dừng hoặc tắt cơ chế theo dõi.
- Không xem hoạt động ứng dụng là bằng chứng người dùng cố tình phớt lờ tin nhắn.

Những sự kiện này chỉ là tín hiệu hành vi chưa đầy đủ, không phải bản ghi toàn diện về hoạt động thiết bị.

## 10. An toàn, quyền riêng tư và kiểm soát quyền hạn

### 10.1. Safety Escalation

Những biểu hiện mệt mỏi hoặc căng thẳng có ý nghĩa chưa rõ ràng không được tự động phân loại thành tình huống khẩn cấp.

Trợ lý cần hỗ trợ việc hỏi làm rõ, đưa ra phản hồi phù hợp ngay khi cần và khuyến khích sự hỗ trợ từ con người trong những tình huống cần thiết.

Việc sử dụng Cloud LLM phải được người dùng đồng ý và thông báo trước, đồng thời chỉ truyền dữ liệu liên quan đến yêu cầu.

Hệ thống không được thể hiện rằng LLM có thể độc lập bảo đảm an toàn cho người dùng.

### 10.2. Bảo mật cấu hình

Những đề xuất cấu hình do LLM tạo ra phải trải qua:

1. Xác minh schema nghiêm ngặt.
2. Chỉ chấp nhận những trường cấu hình nằm trong allowlist.
3. Kiểm tra kiểu dữ liệu.
4. Kiểm tra giới hạn giá trị.
5. Giới hạn tốc độ thay đổi và cooldown.
6. Kiểm tra quyền hạn.
7. Ghi nhật ký và hỗ trợ hoàn nguyên.

Những thay đổi cấu hình nhạy cảm phải được con người phê duyệt rõ ràng.

JSON hợp lệ không đồng nghĩa với việc hệ thống đã được bảo vệ khỏi prompt injection.

Mọi tin nhắn của người dùng và đầu ra do mô hình tạo ra phải được xử lý như dữ liệu chưa đáng tin cậy.

## 11. Công nghệ sử dụng

### Python Prototype

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python |
| Tích hợp Discord | discord.py |
| Cơ sở dữ liệu | SQLite |
| Lập lịch | APScheduler hoặc asyncio |
| Suy luận cục bộ | Ollama |
| Machine Learning | PyTorch |
| Mô hình thử nghiệm | Mamba |
| Cấu hình | JSON / Pydantic |
| Kiểm thử | pytest |

### Rust Runtime trong tương lai

Sau khi prototype Python và những giao diện chính được kiểm chứng:

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ chính | Rust |
| Async Runtime | Tokio |
| Tích hợp Discord | Serenity |
| Serialization | Serde |
| Cơ sở dữ liệu | SQLite |
| HTTP Client | Reqwest |

Quá trình huấn luyện mô hình có thể tiếp tục sử dụng Python.

Inference ban đầu có thể chạy thông qua một dịch vụ Python riêng, thay vì phải lập tức viết lại mô hình bằng Rust.

## 12. Lộ trình thử nghiệm

### Experiment 0 — MVP trong 7 ngày

Thử nghiệm đầu tiên tập trung xác minh quy trình quản lý tiến độ cá nhân.

Phạm vi tối thiểu:

- Discord check-in.
- Lưu dữ liệu bằng SQLite.
- Theo dõi mục tiêu hằng ngày.
- Xác minh tiến độ cơ bản.
- Rule-first routing.
- Gọi Local LLM khi cần.
- Deferred Evaluation.
- Nhận sự kiện cơ bản từ iOS Shortcuts.

Huấn luyện Mamba, On-Policy Distillation và đánh giá LSTM Critic không thuộc phạm vi MVP tối thiểu trong 7 ngày.

### Các chỉ số đánh giá

**Behavioral Metrics — Chỉ số hành vi**

- Tỷ lệ phản hồi check-in.
- Tỷ lệ hoàn thành mục tiêu.
- Mức độ hữu ích do người dùng tự đánh giá.
- Tần suất thông báo không cần thiết.
- Số lần điều chỉnh lịch mang lại lợi ích.

**Routing Metrics — Chỉ số định tuyến**

- Mức độ thống nhất với nhãn tham chiếu đã được đánh giá.
- Tỷ lệ gọi LLM không cần thiết.
- Tỷ lệ bỏ sót những tình huống cần chuyển cấp xử lý.
- Độ trễ suy luận.
- Mức sử dụng CPU và bộ nhớ.

**Learning Metrics — Chỉ số học tập**

- Tỷ lệ vượt qua kiểm tra xác minh.
- Tần suất ôn tập.
- Những dấu hiệu liên quan đến khả năng ghi nhớ hoặc quên kiến thức.

Bảy ngày quan sát cá nhân chưa đủ để kết luận hệ thống có thể cải thiện hành vi một cách tổng quát.

## 13. Nhật ký thử nghiệm

Mỗi thử nghiệm cần ghi nhận:

```text
Experiment ID:
Date:
Hypothesis:
Configuration:
Expected Outcome:
Observed Outcome:
Metrics:
Unexpected Behavior:
Limitations:
New Questions:
Next Iteration:
```

Ý nghĩa:

- **Hypothesis:** Giả thuyết ban đầu.
- **Configuration:** Cấu hình thử nghiệm.
- **Expected Outcome:** Kết quả kỳ vọng.
- **Observed Outcome:** Kết quả thực tế.
- **Metrics:** Các số liệu đo lường.
- **Unexpected Behavior:** Những hành vi ngoài dự kiến.
- **Limitations:** Giới hạn của thử nghiệm.
- **New Questions:** Những câu hỏi mới được phát hiện.
- **Next Iteration:** Hướng điều chỉnh tiếp theo.

Các thay đổi kiến trúc nên được hỗ trợ bởi bằng chứng thực nghiệm, thay vì chỉ dựa trên một lần xảy ra lỗi riêng lẻ.

## 14. Lộ trình phát triển

- [ ] Xây dựng Python accountability prototype.
- [ ] Tích hợp Discord và SQLite.
- [ ] Xây dựng scheduler và hệ thống ưu tiên sự kiện.
- [ ] Tích hợp iOS Shortcuts.
- [ ] Triển khai Adaptive Rule-based Router.
- [ ] Hoàn thành thử nghiệm cá nhân trong 7 ngày.
- [ ] Thiết kế tập dữ liệu hành vi và simulator.
- [ ] Thiết lập các baseline bằng luật và mô hình neural đơn giản.
- [ ] Huấn luyện Tiny Mamba Decision Model bằng supervised learning.
- [ ] Thử nghiệm On-Policy Distillation.
- [ ] So sánh Rule, MLP và LSTM Critic.
- [ ] Chuyển các thành phần runtime ổn định sang Rust.
- [ ] Tích hợp Language Teacher Harness nếu cần.
- [ ] Đánh giá khả năng triển khai lâu dài trên VPS.

## 15. Những câu hỏi nghiên cứu

1. Một Sequential Decision Model nhỏ có thể giảm những lần suy luận LLM không cần thiết hay không?
2. Chính sách dựa trên Mamba có vượt qua những phương pháp đơn giản hơn trong bài toán định tuyến hành vi không?
3. On-Policy Distillation có cải thiện khả năng xử lý những trạng thái phát sinh từ chính chính sách của Student không?
4. Một Critic độc lập có mang lại đủ cải thiện để bù đắp chi phí tính toán bổ sung không?
5. Cơ chế quản lý tiến độ thích ứng có thể tăng mức độ tương tác mà không gây mệt mỏi vì thông báo không?
6. Làm thế nào để hệ thống thích ứng với hành vi mà vẫn bảo vệ quyền tự chủ và riêng tư của người dùng?
7. Những thành phần nào thực sự hưởng lợi khi được chuyển sang Rust?

---

## 16. Trạng thái dự án

README này mô tả kiến trúc dự kiến và lộ trình nghiên cứu.

Hệ thống chưa được xác minh thông qua những thử nghiệm hoàn chỉnh.

Các tuyên bố về hiệu năng mô hình, cải thiện hành vi và khả năng triển khai không được xem là đã chứng minh cho đến khi những thử nghiệm tương ứng được hoàn thành.

### Nguyên tắc phát triển

**Xây dựng hệ thống nhỏ nhất có thể kiểm thử, thu thập bằng chứng và chỉ tăng độ phức tạp khi kết quả thực sự chứng minh điều đó là cần thiết.**

Mục tiêu không phải tối đa hóa số lượng model, agent hay subsystem.

Mục tiêu là nghiên cứu liệu một hệ thống ra quyết định nhỏ, có khả năng thích ứng, có thể hỗ trợ người dùng duy trì cam kết cá nhân trong khi giảm chi phí suy luận, bảo vệ quyền tự chủ và duy trì độ tin cậy trong vận hành hay không.