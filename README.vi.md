# Personal Accountability AI

[English](README.md) | [한국어](README.ko.md) | [Tiếng Việt](README.vi.md)

Trợ lý Discord ưu tiên xử lý cục bộ, đã có bản triển khai hoạt động cho cuộc thi dự án SOL-AIM 2026, với hạn nộp sản phẩm cuối cùng vào ngày 2 tháng 12 năm 2026. Hệ thống quản lý một mục tiêu lặp lại hằng ngày cho một người dùng Discord được chỉ định, gửi tin nhắn hỏi tiến độ theo giờ Hàn Quốc (KST) khi người dùng chủ động đăng ký, phân biệt thời gian tự báo cáo với kết quả kiểm tra kiến thức, và yêu cầu phê duyệt trước khi điều chỉnh mục tiêu.

Trạng thái cá nhân được lưu trong SQLite. Tin nhắn Discord vẫn đi qua nền tảng Discord, nên toàn bộ hệ thống không hoạt động hoàn toàn ngoại tuyến. Tính năng tư vấn bằng Ollama cục bộ là tùy chọn và mặc định bị tắt. Các lệnh `/setgoal`, `/report`, `/summary` và những bản ghi cơ sở dữ liệu hiện có được giữ lại.

## Cài đặt và chạy

Cần Python 3.11.9 trở lên và [uv](https://docs.astral.sh/uv/). Chạy các lệnh sau bằng PowerShell tại thư mục gốc của kho mã:

```powershell
uv sync --locked
Copy-Item .env.example .env
# Chỉnh sửa .env trên máy rồi chạy:
uv run python bot.py
```

Không ghi đè nếu đã có tệp `.env`. Bot kiểm tra cấu hình trước khi kết nối. Việc import module `bot` không tự tạo cơ sở dữ liệu, đọc `.env` hay kết nối Discord. Chỉ chạy một tiến trình bot cho mỗi cơ sở dữ liệu.

Nếu không có quyền truy cập thư mục cache mặc định của uv, có thể dùng cache ngay trong thư mục dự án:

```powershell
uv --cache-dir .uv-cache sync --locked
uv --cache-dir .uv-cache run python bot.py
```

## Cấu hình

| Biến môi trường | Bắt buộc | Giá trị mặc định / ý nghĩa |
|---|---|---|
| `DISCORD_TOKEN` | Có | Token của bot; phải giữ bí mật |
| `ALLOWED_USER_ID` | Có | ID người dùng Discord được phép truy cập, là số nguyên dương |
| `DISCORD_GUILD_ID` | Không | `0`; có thể chỉ định ID máy chủ dùng để phát triển |
| `DATABASE_PATH` | Không | `data/accountability.db`, tính từ thư mục làm việc khi chạy |
| `MIDDAY_CHECKIN` | Không | `12:00`, múi giờ `Asia/Seoul`, định dạng 24 giờ `HH:MM` |
| `EVENING_REVIEW` | Không | `21:00`, múi giờ `Asia/Seoul` |
| `OLLAMA_ENABLED` | Không | `false`; chỉ nhận `true` hoặc `false` |
| `OLLAMA_URL` | Không | `http://127.0.0.1:11434`; chỉ chấp nhận địa chỉ HTTP cục bộ |
| `OLLAMA_MODEL` | Không | `qwen3:0.6b`; cần cài riêng mô hình cục bộ muốn sử dụng |
| `OLLAMA_TIMEOUT` | Không | `10` giây; phạm vi cho phép từ 0.1 đến 60 giây |

Các tên cũ `DISCORD_BOT_TOKEN` và `GUILD_ID` vẫn được chấp nhận nếu biến chính thức tương ứng chưa được đặt. Giờ mặc định trong biến môi trường được áp dụng khi bật check-in; sau khi khởi động lại, hệ thống tiếp tục dùng lựa chọn đã lưu của người dùng. Hệ thống sử dụng múi giờ `Asia/Seoul`; trên Windows không có cơ sở dữ liệu múi giờ, hệ thống dùng UTC+09:00 thay thế, tương ứng với giờ Hàn Quốc hiện tại.

## Thiết lập Discord

1. Tạo ứng dụng và bot trong Discord Developer Portal. Điền token của bot vào `.env` trên máy.
2. Bật Developer Mode trong Discord, sao chép ID của chính bạn và điền vào `ALLOWED_USER_ID`. Nếu cần, điền ID máy chủ phát triển vào `DISCORD_GUILD_ID`.
3. Cài bot vào máy chủ với hai phạm vi `bot` và `applications.commands`. Cho phép sử dụng lệnh ứng dụng và nhận DM từ bot; không cần cấp quyền quản trị viên.
4. Khởi động bot. Bot đồng bộ lệnh toàn cục để sử dụng trong máy chủ và DM, đồng thời đồng bộ lệnh riêng cho máy chủ phát triển nếu đã cấu hình. Có thể cần chờ một thời gian để lệnh toàn cục xuất hiện trên Discord.
5. Mọi lệnh đều kiểm tra ID người dùng đã cấu hình. Phải chủ động bật check-in. Nếu Discord chặn DM, sau lần thử gửi, `/checkins action:status` sẽ hiển thị `blocked`; hãy cho phép DM trước khung giờ gửi tiếp theo.

Người dùng được phép có thể nhắn riêng cho bot, chẳng hạn “Tiến độ hôm nay của tôi?”. Không cần các privileged intent Message Content, Members hay Presence. Phản hồi lệnh chỉ hiển thị với người thực hiện. Tin nhắn theo lịch được gửi qua DM khi người dùng đã đăng ký. Xem ví dụ tiếng Việt và các bước xác nhận, phê duyệt trong [hướng dẫn thử DM](docs/DISCORD_DM_DEMO.md).

## Lệnh và kịch bản trình diễn

Có thể trình diễn bằng một mục tiêu học tiếng Hàn giả lập. Giữ nguyên tên tùy chọn tiếng Anh như Discord hiển thị khi dùng lệnh slash.

1. `/setgoal title:Korean minutes:30` đặt mục tiêu học 30 phút mỗi ngày.
2. `/checkins action:enable midday:12:00 evening:21:00` bật check-in theo KST. Để trình diễn nhanh, đặt giờ `midday` sau thời điểm hiện tại vài phút và giờ `evening` sau đó. Giữ bot chạy và chờ DM.
3. `/report minutes:30` ghi nhận **tổng số phút của hôm nay**, thay thế tổng hiện tại nhưng vẫn giữ từng sự kiện báo cáo trước đó. Các lần báo cáo không được cộng dồn với nhau.
4. `/quiz` đưa ra câu hỏi từ vựng tiếng Hàn đã định nghĩa sẵn. Làm theo hướng dẫn `/answer attempt:<id> answer:<answer>`. Câu hỏi minh họa yêu cầu nghĩa tiếng Anh của 물; đáp án là `water`.
5. Thử trả lời sai để thấy trạng thái `review_required`, sau đó dùng `/quiz` và trả lời đúng để thấy `passed_after_support`. Trả lời đúng ngay lần đầu cho trạng thái `verified`. Trạng thái chỉ áp dụng cho từ vựng được kiểm tra và phiên bản báo cáo hiện tại; không phải bằng chứng độc lập rằng người dùng đã học 30 phút. Một báo cáo mới bắt đầu với trạng thái xác minh `reported`.
6. `/recover reason:Unexpected work shift` thông báo trở ngại như ca làm đột xuất. Bot đề xuất giảm một nửa chỉ tiêu hôm nay nhưng chưa thay đổi chỉ tiêu.
7. `/adjust proposal:<id> decision:approve` áp dụng đề xuất cho hôm nay. Chọn `reject` hoặc `cancel` sẽ giữ nguyên chỉ tiêu. Mục tiêu lặp lại 30 phút và chỉ tiêu ban đầu của ngày vẫn được lưu. Ngày mai hệ thống quay về chỉ tiêu của mục tiêu lặp lại.
8. `/summary` hiển thị chỉ tiêu ban đầu, chỉ tiêu đang áp dụng, số phút tự báo cáo, kết quả kiểm tra và các trở ngại. DM tổng kết buổi tối sử dụng cùng nội dung này.
9. Dừng rồi khởi động lại bot, sau đó kiểm tra `/summary`, `/history` và `/checkins action:status`. Trạng thái được giữ lại và khung check-in trong ngày đã bắt đầu xử lý gửi sẽ không được gửi lại.
10. Dùng `/checkins action:pause`, `/checkins action:resume` hoặc `/checkins action:disable` để tạm dừng, tiếp tục hoặc tắt nhắc nhở. `/ask text:<question>` yêu cầu tư vấn cục bộ tùy chọn. `/ask ... non_urgent:true` ghi quyết định định tuyến `DEFER` mà không gọi mô hình; hãy tự gửi lại câu hỏi khi muốn thảo luận.

## Kiến trúc và lưu trữ dữ liệu

| Module | Trách nhiệm |
|---|---|
| `core.py` | API GoalStore hiện có, ảnh chụp kế hoạch từng ngày, lịch sử phiên bản báo cáo, tổng kết hằng ngày |
| `memory.py` | Migration SQLite theo giao dịch, sự kiện chỉ ghi thêm, truy vấn lịch sử theo điều kiện |
| `bot.py` | Lệnh slash có kiểm tra quyền, phản hồi riêng tư, adapter DM và vòng lặp nền |
| `scheduler.py` | Cấu hình đăng ký check-in, lưu trạng thái các khung gửi hằng ngày, phục hồi sau khởi động lại và ghi nhận kết quả gửi |
| `verification.py` | Câu hỏi từ vựng định nghĩa sẵn và bảo tồn lịch sử các lần kiểm tra |
| `recovery.py` | Đề xuất điều chỉnh hằng ngày theo quy tắc, phê duyệt/từ chối/hủy trong giao dịch nguyên tử |
| `router.py` | Giao diện Router có thể thay thế, quy tắc RULE/CLARIFY/LLM/DEFER và nhật ký quyết định |
| `llm_adapter.py` | Suy luận HTTP cục bộ có giới hạn, kiểm tra JSON nghiêm ngặt và phản hồi dự phòng |
| `config.py`, `time_utils.py` | Kiểm tra cấu hình môi trường, xử lý ngày, số phút và giờ KST |
| `experiments/router/` | Dữ liệu tổng hợp, huấn luyện mô hình cơ sở và quy trình đánh giá tách biệt với ứng dụng |

Migration dùng `PRAGMA user_version` của SQLite, chạy trong giao dịch và thêm bảng mà không xóa bản ghi hiện có. Hai bảng gốc `goals` và `reports` vẫn tương thích. Migration bổ sung lịch sử phiên bản báo cáo và ảnh chụp kế hoạch ngày từ các bản ghi cũ. Thời điểm tạo báo cáo cũ được đánh dấu là không xác định; những lần đổi mục tiêu chưa từng được lưu không thể khôi phục, nên chỉ tiêu khi chuyển đổi dựa trên giá trị mục tiêu đang có. Trước khi nâng cấp, hãy dừng bot rồi sao lưu cơ sở dữ liệu; giữ tệp `.db` riêng tư.

`summary()['status']` giữ nguyên ý nghĩa trạng thái hoạt động (`not_reported` hoặc `reported`); `verification_state` thể hiện riêng năm trạng thái xác minh. Những lần kiểm tra trước vẫn tồn tại trong cơ sở dữ liệu sau khi có báo cáo hoặc lần kiểm tra mới. Khi kế hoạch đã thay đổi, hệ thống không cho phê duyệt đề xuất cũ; việc phê duyệt cần thao tác rõ ràng của người dùng trong chính ngày đó. `Memory(store).history(user_id, day=None, kind=None, limit=20)` trả về các sự kiện liên quan từ mới nhất đến cũ nhất, kèm nguồn thông tin và dấu thời gian.

## Độ tin cậy và phạm vi hoạt động của bộ lập lịch

- Kiểm tra lịch mỗi 30 giây sau khi Discord sẵn sàng. Không đăng ký thì không có DM theo lịch.
- Bản ghi giữ quyền xử lý duy nhất theo `(user_id, ngày KST, khung giờ)` ngăn gửi trùng trong ngày, kể cả khi nhiều lượt kiểm tra chạy đồng thời hoặc bot khởi động lại.
- Sau khi khởi động lại, bot có thể xử lý bù lịch hợp lệ của hôm nay trong vòng sáu giờ kể từ giờ hẹn. Lịch của ngày cũ và những khung giờ đã qua trước lúc bật, cấu hình lại hoặc tiếp tục sẽ bị bỏ qua. Không gửi dồn các thông báo tồn đọng.
- Trạng thái bắt đầu xử lý được ghi và commit trước khi gửi qua Discord. Nếu tiến trình dừng bất thường, kết quả gửi có thể không xác định. Bản ghi còn ở trạng thái đang gửi quá hai phút được chuyển thành `unknown` và không tự động gửi lại. Cách này ưu tiên hạn chế DM trùng nhưng không bảo đảm mọi DM đều đến nơi. Không bảo đảm gửi đúng một lần trên cả Discord và SQLite.
- Với lỗi chắc chắn chưa gửi thành công và có thể thử lại, hệ thống thử tối đa ba lần, cách nhau ít nhất năm phút. Không có quyền gửi hoặc không tìm thấy người nhận sẽ thành `blocked`. Lỗi HTTP của Discord, hết thời gian chờ và các kết quả không chắc chắn khác sẽ thành `unknown`, không tự động thử lại.
- Tạm dừng hoặc tắt ngăn bắt đầu các lượt gửi mới. Lượt gửi đang diễn ra vẫn có thể hoàn tất. Đổi giờ không mở lại khung đã được giữ quyền xử lý trong ngày. Bật lại hoặc tiếp tục chỉ áp dụng cho các giờ hẹn sắp tới.
- Không phản hồi hoặc gửi thất bại không bao giờ là căn cứ để kết luận người dùng trì hoãn, thiếu trung thực hay thất bại với mục tiêu.

## Tư vấn bằng Ollama cục bộ (tùy chọn)

Cài riêng Ollama và mô hình cục bộ muốn dùng, đặt `OLLAMA_ENABLED=true`, rồi sử dụng `/ask`. Adapter gọi [`POST /api/generate`](https://docs.ollama.com/api/generate) với chế độ streaming bị tắt và một JSON schema rõ ràng. Yêu cầu có giới hạn tổng thời gian và kích thước phản hồi; chuyển hướng và proxy từ biến môi trường bị vô hiệu hóa. Chỉ chấp nhận địa chỉ loopback.

Đầu ra chỉ được chứa đúng một hành động trong danh sách cho phép (`SUGGEST` hoặc `ASK_FOLLOWUP`) và một thông điệp ngắn. Nội dung được hiển thị rõ là đề xuất của mô hình. Không có bộ thực thi đầu ra mô hình, và mô hình không thể thay đổi mục tiêu, lịch, cài đặt riêng tư, tệp hay thực thi lệnh. Kiểm tra JSON không bảo đảm lời khuyên đúng hoặc phù hợp. Khi lỗi, hết thời gian chờ, JSON không hợp lệ hoặc backend bị tắt, hệ thống trả lời tiếp theo bằng quy tắc có sẵn. Theo dõi mục tiêu và lập lịch cơ bản không cần Ollama. Nội dung tư vấn và prompt gốc không được lưu; chỉ lưu kết quả gọi và độ trễ.

## Kiểm thử và phát triển

```powershell
uv run python -m pytest -q
uv run python check.py
uv run python -m compileall -q bot.py core.py config.py scheduler.py router.py verification.py recovery.py memory.py llm_adapter.py time_utils.py
# Bao gồm kiểm thử lưu, tải lại và đánh giá mô hình:
uv run --extra research python -m pytest -q
```

Kiểm thử dùng cơ sở dữ liệu SQLite tạm chứa dữ liệu tổng hợp và mock Discord/Ollama. Không cần token, bot đang chạy, mô hình cục bộ hay dịch vụ bên ngoài. Nếu chưa cài phần phụ thuộc bổ sung `research`, kiểm thử đánh giá mô hình sẽ được bỏ qua. Ba kiểm thử lõi ban đầu vẫn được giữ lại. Phạm vi bao gồm migration, lưu trữ bền vững, đầu vào không hợp lệ, kiểm tra quyền của mọi lệnh, thử lại/chống trùng/khởi động lại/tạm dừng lịch, chuyển trạng thái xác minh, ngăn phê duyệt đề xuất cũ và toàn bộ luồng lệnh đầu cuối.

## Thử nghiệm mô hình ngoại tuyến

```powershell
uv sync --locked --extra research
uv run --extra research python -m experiments.router.run --seed 42
# Có thể chọn thư mục đầu ra và kích thước dữ liệu:
uv run --extra research python -m experiments.router.run --seed 42 --samples-per-family 20 --output experiments/router/artifacts
```

Sau khi cài các phụ thuộc, quá trình chạy không cần mạng. Thử nghiệm ghi dữ liệu tổng hợp huấn luyện/validation/test dạng JSONL, `baseline.joblib` và `metrics.json`. Chỉ số gồm precision/recall/F1 theo lớp, ma trận nhầm lẫn, trung vị và p95 của độ trễ suy luận từng đầu vào, kích thước tệp mô hình, thời gian huấn luyện và phép đo giới hạn về cấp phát bộ nhớ Python. Chỉ tải những tệp joblib do chính bạn tạo và tin cậy.

Mặc định chia 240/80/160 mẫu từ các nhóm mẫu kịch bản không giao nhau. Chỉ dùng tập validation để chọn siêu tham số. Cả hai bộ định tuyến nhận cùng văn bản và metadata tường minh; mã nhóm kịch bản và nhãn không được đưa vào đặc trưng của mô hình. Xem [quy trình và kết quả thử nghiệm](experiments/router/README.md). Mô hình cơ sở đã đánh giá không được áp dụng vào bot đang chạy.

## Đã hoàn thành, thử nghiệm và kế hoạch tiếp theo

MVP đã hoàn thành: các lệnh gốc, lưu bền vững báo cáo tiến độ, DM check-in theo KST khi người dùng đăng ký, tạm dừng/tiếp tục, tổng kết hằng ngày, kiểm tra từ vựng, điều chỉnh mục tiêu trong ngày sau phê duyệt, lịch sử, định tuyến theo quy tắc và tư vấn cục bộ tùy chọn có phản hồi dự phòng.

Hạ tầng nghiên cứu đã hoàn thành: sinh dữ liệu tổng hợp, chia tập theo nhóm kịch bản, huấn luyện TF-IDF/logistic regression, lưu và tải lại mô hình, chỉ số phân loại, ma trận nhầm lẫn và so sánh với quy tắc. Kết quả tổng hợp chỉ đánh giá tập dữ liệu định tuyến này, không chứng minh hiệu quả hỗ trợ duy trì cam kết ở người dùng thực tế.

Nghiên cứu tiếp theo: bổ sung kịch bản định tuyến và đặc trưng metadata đa dạng đã được rà soát; xác định bộ mô phỏng theo thời gian và quy ước đầu vào/đầu ra chuỗi trước khi so sánh LSTM hoặc Mamba; sau đó mới xem xét distillation hay critic với mục tiêu huấn luyện có cơ sở. Chưa có mô hình LSTM, Mamba, critic hay distillation nào được huấn luyện hoặc đánh giá. Embedding, tìm kiếm vector, suy luận đám mây, theo dõi iOS, hàng đợi ưu tiên/trì hoãn tổng quát và triển khai công khai nằm ngoài MVP này. Hiện `DEFER` chỉ đề nghị hoãn thủ công; chưa có worker tự động xử lý suy luận trì hoãn.

Giới hạn quyền riêng tư: SQLite chưa được mã hóa và chưa có giao diện tự động quản lý thời hạn lưu hoặc xóa dữ liệu cá nhân. Người có quyền truy cập hệ thống tệp cục bộ có thể đọc dữ liệu. Discord lưu giữ những tin nhắn được truyền qua nền tảng. Quyền truy cập của một người dùng được kiểm tra tại lớp tích hợp Discord; thư viện nghiệp vụ là API cục bộ dành cho bên gọi đáng tin cậy. Không tự động thu thập hội thoại riêng tư để huấn luyện. Log không chứa nội dung tin nhắn, bản ghi cá nhân hay token. Thông tin bí mật, cơ sở dữ liệu, cache và sản phẩm thực nghiệm được loại khỏi theo dõi của Git. Trợ lý không chẩn đoán tình trạng sức khỏe hay suy đoán ý định từ tốc độ phản hồi.

Bản đề xuất kiến trúc tiếng Việt trước đây được giữ trong [docs/RESEARCH_ROADMAP.vi.md](docs/RESEARCH_ROADMAP.vi.md); bản tiếng Anh ở [docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md). Đây là những ý tưởng cho tương lai, không phải danh sách tính năng đã triển khai. Bằng chứng triển khai và kiểm chứng nằm trong [docs/IMPLEMENTATION_LOG.md](docs/IMPLEMENTATION_LOG.md).
