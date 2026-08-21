# راهنمای فهم فصل ۲: The Financial Data Universe

> خلاصهٔ آموزشی فصل ۲ از *Machine Learning for Trading, 3rd Edition* (Stefan Jansen).
> هدف: قبل از ساخت استراتژی بفهمی چه داده‌ای هست، کجا خراب می‌شود، و چطور ذخیره/اعتبارسنجی‌اش کنی.

## پیام اصلی فصل (یک جمله)

پیچیدگی داده‌های مالی فرصت می‌سازد، ولی بدون انضباط در timestamp، adjustment، شناسه و revision، backtestها **بی‌صدا و اشتباه** می‌شوند — نه با خطای واضح، بلکه با alpha ساختگی.

```text
Taxonomy (2.1) → Asset classes (2.2) → Due diligence (2.3) → Storage (2.4)
     │                  │                    │
     ├─ Market          ├─ Equities/Futures   ├─ PIT
     ├─ Fundamental     ├─ FX / OTC           ├─ Survivorship
     └─ Alternative     └─ Crypto/Options     ├─ Corporate actions
                                              └─ Identifiers
```

Notebookهای مرتبط در همین پوشه با شمارهٔ بخش کتاب هم‌ترازند؛ انتهای هر بلوک لینک کوتاه دارد.

---

## بلوک ۱ — Taxonomy و چهار قفل metadata (§2.1)

### ایدهٔ اصلی

یک dataset فقط «اعداد» نیست؛ یک **روش اندازه‌گیری** است با قوانین داخلی: timestamp، تقویم معاملاتی، adjustment، شناسه، revision. وقتی «قیمت روزانه» دانلود می‌کنی، داری تعریف vendor را می‌پذیری — نه حقیقت مطلق بازار را.

### سه دسته داده

| نوع | چه چیزی را اندازه می‌گیرد | نکته مهندسی کلیدی |
|-----|---------------------------|-------------------|
| **Market** | معاملات، quotes، OHLC | هرچه aggregation بالاتر، مدل ساده‌تر؛ اطلاعات عرضه/تقاضا کمتر |
| **Fundamental** | محرک‌های ارزش (صورت‌های مالی، موجودی، نرخ بهره، on-chain) | **release-time ≠ event-time**؛ سؤال درست: «در لحظه تصمیم چه می‌دانستیم؟» |
| **Alternative** | پروکسی‌های خارج از feeds استاندارد (ماهواره، کارت اعتباری، sentiment، ESG) | تنوع بالا + بار validation سنگین (coverage، selection، حقوق استفاده) |

**Market data — سلسله‌مراتب aggregation**

- پایه: event خام (order/execution) در venue
- بالا: barهای OHLCV در بازهٔ زمانی
- «بهترین قیمت» به پوشش venue و قانون ادغام feed بستگی دارد
- feed رایگان اغلب تأخیری/فیلترشده است؛ حتی قیمت «ساده» نیاز به validation دارد

**Fundamental — دو دام مهندسی**

1. داده با تأخیر انتشار، embargo و revision می‌آید
2. تعریف‌ها را قوانین نهادی می‌سازند (حسابداری، آژانس‌ها، پروتکل کریپتو)

**Alternative — سؤالات validation**

- coverage در طول زمان پایدار است؟
- selection effect در نمونه چیست؟
- روش قابل بازتولید است؟
- usage rights برای backtest/live/training اجازه می‌دهد؟

### چهار چیزی که قبل از مدل باید قفل کنی

1. **Timestamp semantics** — close یعنی last trade، auction، یا snapshot؟ کدام timezone؟
2. **Corporate-action methodology** — split/dividend نقطه به نقطه یا back-adjusted؟
3. **Identifier stability** — ticker کافی نیست؛ نگاشت با تاریخ اعتبار
4. **Revisions/restatements** — نسخهٔ «امروز» یا نسخهٔ «آن موقع قابل‌دانستن»؟

این انتخاب‌ها را در config/metadata پایپ‌لاین نگه دار تا downstream دوباره تفسیرشان نکند.

### چک فهم — بلوک ۱

**سؤال:** اگر دیروز close اپل را از دو منبع بگیری و یکی auction close باشد و دیگری last trade، چرا ممکن است هر دو «درست» باشند ولی برای مدل تو ناسازگار؟

**جواب:** هر دو تعریف معتبر از «close»اند، ولی زمان و مکان کشف قیمت فرق دارد. اگر بدون قفل کردن تعریف در metadata آن‌ها را مخلوط کنی، featureها و labelها روی مقیاس زمانی ناسازگار سوار می‌شوند و bias خاموش می‌گیری.

---

## بلوک ۲ — کلاس دارایی = ساختار بازار = کیفیت داده (§2.2)

الگوی تکرارشونده در هر کلاس: **چه می‌بینی → حالت‌های شکست → تصمیم مهندسی**.

| کلاس | مشاهده‌پذیری | شکست رایج | اولویت مهندسی |
|------|--------------|-----------|----------------|
| Equities | بالا | corp actions، fragmentation | تعریف close + adjustment |
| Futures | بالا | roll و continuity | سری continuous صریح |
| Options | بالا ولی دم نازک | نویز illiquid | سطح IV، نه کل chain |
| Crypto | متوسط | volume جعلی، 24/7 | غربال venue |
| FX | پایین (OTC) | close convention | قانون aggregation |
| Fixed income / Swaps | پایین | matrix pricing، curve | جدا کردن quote واقعی از مدل |
| Commodities | متوسط | spot مبهم، delivery | metadata قرارداد |

**نکته:** ابزارهای order-book فصل ۳ روی exchange-traded جواب می‌دهند؛ روی OTC و DEX-AMM نه.

### Equities (تمرکز)

- بازار چند venue (lit + dark/ATS)؛ در آمریکا NBBO وجود دارد
- «close» اغلب یعنی **auction close**، نه last trade
- شکست‌ها: fragmentation timestamp، corporate actions، ADR/cross-list، تبدیل ارز
- تصمیم‌ها قبل از research: تعریف close، consolidated vs direct feed، total-return methodology، سیاست شناسه

**Notebook:** [`01_us_equities_eda`](01_us_equities_eda.ipynb)، [`02_corporate_actions`](02_corporate_actions.ipynb)

### Futures (تمرکز)

- داده per-contract شفاف است، ولی **ticker دائمی وجود ندارد**
- باید continuous series بسازی؛ roll rule (calendar / volume / OI) و روش continuity (ratio / difference) انتخاب backtest-defining هستند
- سری back-adjusted معمولاً شبیه excess return است مگر collateral را جدا مدل کنی
- تصمیم: تاریخچهٔ خام قراردادها را نگه دار + یک یا چند continuous variant

**Notebook:** [`04_cme_futures_eda`](04_cme_futures_eda.ipynb)، [`06_futures_continuous`](06_futures_continuous.ipynb)

### FX (تمرکز)

- عمدتاً OTC؛ consolidated tape ندارد
- «daily close» (مثلاً 5pm NY) یک **convention** است، نه رویداد صرافی
- 4pm London و 5pm NY برای EUR/USD در روزهای پرنوسان چند pip فرق می‌کنند
- تصمیم: مشخص کن کدام قیمت واقعاً قابل معامله است (best-of، VW-mid، یک venue خاص) و close convention را در metadata بنویس

**Notebook:** [`12_fx_pairs_eda`](12_fx_pairs_eda.ipynb)

### یادداشت کوتاه روی بقیه

- **ETP/ETF:** لایهٔ دوم داده لازم است (NAV، holdings، premium/discount). holdings امروز را برای دیروز استفاده نکن → lookahead.
- **Options:** سطح IV/skew/term structure را به‌عنوان dataset طراحی کن، نه خامِ کل chain.
- **Crypto:** CEX شبیه equities؛ DEX با AMM عمق را از reserves می‌گیرد نه order book. Venue screening اجباری است.
- **Fixed income / swaps / commodities:** اغلب quote indicative یا مدل‌شده؛ «price» را صریح تعریف کن.

### چک فهم — بلوک ۲

**سؤال:** چرا در futures «قیمت پیوسته» یک حقیقت بازار نیست و یک تصمیم مهندسی است؟

**جواب:** بازار دنباله‌ای از قراردادهای منقضی‌شونده است. برای داشتن یک سری زمانی باید بگویی کی roll می‌کنی و چطور gap را پر/تعدیل می‌کنی. دو قانون roll مختلف دو تاریخچهٔ بازده متفاوت می‌سازند.

**سؤال:** چرا در FX نمی‌توانی فرض کنی یک «قیمت رسمی روزانه» وجود دارد؟

**جواب:** چون بازار OTC است و close فقط قرارداد زمانی بین شرکت‌کنندگان است؛ بدون تعیین venue و قانون aggregation، سری‌هایت قابل مقایسه نیستند.

---

## بلوک ۳ — چهار شکست که alpha جعلی می‌سازند (§2.3)

بسیاری از شکست‌های استراتژی ML در production به مدل برنمی‌گردند؛ به **نقص داده** برمی‌گردند.

### ابعاد عمومی کیفیت داده

Timeliness · Completeness · Accuracy · Consistency · Validity

**Notebook:** [`13_data_quality_framework`](13_data_quality_framework.ipynb)

### ۱) Point-in-time (PIT)

| مفهوم | معنی |
|-------|------|
| Event time | رویداد اقتصادی رخ داد (مثلاً پایان Q4 2019) |
| Knowledge time | اطلاعات عمومی شد (acceptance زمان filing) |
| As-of time | لحظهٔ تصمیم که در backtest بازسازی می‌کنی |
| Reference period | مقدار به چه دوره‌ای ارجاع می‌دهد |

**قانون عملیاتی:** فقط رکوردی با `available_at <= decision_time` مجاز است. Join شناسه‌ها هم باید در بازهٔ اعتبار باشند: `effective_date <= decision_time < end_date`.

بدون **bitemporal storage** و as-of query، restatementها lookahead می‌سازند.

مثال کتاب: revenue Q4 2019 ابتدا ۹۱.۸B گزارش شد؛ بعد از restatement vendor امروز ۹۲.۳B نشان می‌دهد. استفاده از عدد امروز در backtest سال ۲۰۲۰ = lookahead.

چک‌لیست PIT سه لایه دارد: **values**، **universe membership**، **identifier mappings**.

**Notebook:** [`14_point_in_time_validation`](14_point_in_time_validation.ipynb)

### ۲) Survivorship bias

- جهان تاریخی را با «امروز هنوز زنده است» فیلتر نکن
- حذف ورشکستگی‌ها → بازده بیش‌برآورد
- حذف اهداف M&A → گاهی کم‌برآورد (پریمیوم خرید از دست می‌رود)
- استفاده از constituents امروزِ S&P 500 برای backtest تاریخی = فیلتر موفقیت گذشته

در پنل US equities کتاب: از ۳۱۹۹ سهام، ۷۷۷ تا (۲۴.۳٪) قبل از پایان نمونه delist شدند؛ bias تخمینی روی ۲۰۱۴–۲۰۱۸ حدود ۳.۵ تا ۱۵.۳ نقطه درصد بود.

**علاج:** universe نقطه به نقطه در زمان + delisting returns درست (مثلاً CRSP-style).

**Notebook:** [`15_survivorship_bias_detection`](15_survivorship_bias_detection.ipynb)

### ۳) Corporate actions

- split ۷:۱ قیمت خام را ~۸۶٪ «سقوط» نشان می‌دهد
- مثال AAPL در متن: قیمت خام ≈ ۵.۹× بازده تجمعی؛ adjusted ≈ ۳۹۸× → کم‌برآورد حدود ۶۸×
- vendorها در total vs price return و timingِ عامل‌ها فرق دارند؛ مخلوط‌کردنشان خودش lookahead است

**علاج:** متدولوژی را document و validate کن.

**Notebook:** [`02_corporate_actions`](02_corporate_actions.ipynb)

### ۴) Identifier integrity

- ticker reuse: join «موفق» روی XYZ ولی دو شرکت متفاوت → correlation ساختگی
- CUSIP بعد از merger عوض می‌شود؛ vendor ID لبه‌های خاص دارد

**علاج:** permanent ID + بازهٔ تاریخ اعتبار (FIGI/ISIN/CUSIP via crosswalk).

**Notebook:** [`16_provider_comparison`](16_provider_comparison.ipynb)، [`17_complete_pipeline`](17_complete_pipeline.ipynb)

### Vendor due diligence — سه محور

1. **کیفیت:** PIT، survivorship، adjustment، coverage، شناسه
2. **حقوقی:** MNPI، privacy (GDPR/CCPA)، usage rights، auditability
3. **فنی/تجاری:** SLA، نسخه schema، export/portability، rate limit، backfill

**حداقل validation بعد از انتخاب vendor**

- بدون zero-lag برای دادهٔ تأخیری
- universe تاریخی با اطلاعات آینده فیلتر نشده
- adjustment documented
- join روی permanent ID + effective dates
- timezone و session صریح
- outlier/stale quote detection
- versioning و as-of reproducible

**Build vs buy:** پایپ‌لاین، validation، شناسهٔ canonical و transforms استراتژی‌محور را بساز؛ تاریخچهٔ تمیز بلندمدت، crosswalk شناسه، و universe survivorship-aware را در صورت هزینهٔ بالا بخر.

### چک فهم — بلوک ۳

**سؤال:** PIT و survivorship چگونه backtest را خراب می‌کنند؟

**جواب:** PIT وقتی تاریخچه را با اطلاعات بعدی بازنویسی می‌کند، مدل را روی دانش آینده آموزش می‌دهد (lookahead). Survivorship وقتی فقط نام‌های «باقی‌مانده» را نگه می‌دارد، توزیع بازده را منحرف می‌کند — معمولاً به سمت بالا اگر ورشکستگی‌ها حذف شوند. هر دو بدون crash واضح، alpha جعلی می‌سازند.

---

## بلوک ۴ — ذخیره‌سازی و جمع‌بندی (§2.4–2.5)

### اصل انتخاب storage

بهترین فرمت/موتور مطلق وجود ندارد؛ به volume، الگوی دسترسی (scan، point lookup، range، join)، concurrency و بلوغ عملیاتی بستگی دارد.

### File formats (پیش‌فرض research)

| Format | نقش |
|--------|-----|
| **Parquet** | پیش‌فرض persistent: فشرده، columnar، tool support وسیع |
| Arrow IPC / Feather | interchange کوتاه‌عمر / memory-map؛ نه data lake |
| HDF5 | میراث علمی؛ برای بسیاری از scanها از Parquet عقب‌تر |
| CSV | فقط export/interop |

Lakehouse (Delta/Iceberg/Hudi) را وقتی اضافه کن که update سنگین، schema evolution قوی، یا multi-writer لازم است.

**Notebook:** [`20_storage_benchmark_file`](20_storage_benchmark_file.ipynb)

### معماری پیش‌فرض پیشنهادی کتاب

1. داده خام/تمیز: **Parquet پارتیشن‌شده**
2. SQL analytics روی فایل: **DuckDB**
3. feature engineering: **Polars**
4. سرور DB فقط وقتی concurrency/governance/SLA لازم است

| هدف | پیش‌فرض قوی |
|-----|-------------|
| Research velocity | Parquet + Polars |
| SQL روی فایل | DuckDB + Parquet |
| Production reliability | PostgreSQL / TimescaleDB |
| Extreme time-series | kdb+ / ClickHouse |
| High-throughput ingest | QuestDB / ClickHouse |
| ASOF در حافظه | Polars (سپس pandas) |
| ASOF با SQL | DuckDB (سپس QuestDB) |

**ASOF join:** برای هم‌تراز کردن trade با آخرین quote؛ ورودی باید sorted باشد. در مقیاس‌های کتاب، in-memory (Polars) معمولاً از DuckDB SQL سریع‌تر است — با caveat گرم/سرد بودن cache و اینکه sort بخشی از بنچمارک باشد یا نه.

**Notebook:** [`21_storage_benchmark_database`](21_storage_benchmark_database.ipynb)، [`22_pandas_polars_benchmark`](22_pandas_polars_benchmark.ipynb)، [`18_data_management`](18_data_management.ipynb)

### جمع‌بندی فصل (§2.5)

1. با taxonomy (market / fundamental / alternative) dataset را به افق استراتژی وصل کن و فرض‌های timestamp/adjustment/ID/revision را سطحی کن.
2. چهار نقص پیش‌فرض را جدی بگیر: PIT، survivorship، corporate actions، identifier joins.
3. ساختار بازار محدود می‌کند چه چیزی قابل مشاهده است؛ انتخاب vendor = مدیریت ریسک (کیفیت + حقوقی + عملیاتی).
4. برای اکثر research: partitioned Parquet + Polars/DuckDB؛ دیتابیس تخصصی فقط با توجیه عملیاتی.

**پل به فصل ۳:** جزئیات market data، order-book، bar sampling، و محیط نهادی صرافی‌ها — ورودی label/feature و شبیه‌سازی عملکرد تاریخی.

### چک فهم — بلوک ۴

**سؤال:** چرا Parquet + Polars/DuckDB نقطهٔ شروع معقول است؟

**جواب:** اکثر workflowهای research با فایل شروع می‌شوند؛ Parquet فشرده و columnar است و selective read می‌دهد. Polars برای feature engineering سریع است و DuckDB SQL را مستقیم روی همان فایل‌ها می‌آورد. سرور DB را فقط وقتی concurrency و governance هزینهٔ عملیاتی را توجیه می‌کنند اضافه می‌کنی.

---

## خروجی مورد انتظار — خودآزمایی نهایی

با زبان خودت جواب بده:

1. تفاوت market / fundamental / alternative چیست؟
2. چرا «close» و «قیمت» در FX / futures / ETF معانی متفاوت دارند؟
3. PIT و survivorship چگونه backtest را خراب می‌کنند؟
4. چرا Parquet + Polars/DuckDB نقطهٔ شروع معقول است؟

### پاسخ‌های مرجع کوتاه

1. **Market** رفتار معامله را ثبت می‌کند؛ **fundamental** محرک‌های ارزش اقتصادی را (با تأخیر انتشار)؛ **alternative** سیگنال‌های خارج از feeds استاندارد است که اغلب پروکسی fundamentalsاند ولی validation سخت‌تری می‌خواهند.
2. چون ساختار بازار فرق دارد: FX فقط convention دارد؛ futures هویت قرارداد در زمان می‌شکند و continuous ساخته می‌شود؛ ETF هم قیمت ثانویه دارد هم NAV/holdings با ریسک premium و look-through leakage.
3. PIT دانش آینده را به گذشته می‌چسباند؛ survivorship جهان را با نتیجهٔ آینده فیلتر می‌کند — هر دو alpha جعلی می‌سازند.
4. چون تعادل خوبی بین اندازه، سرعت خواندن، و سادگی عملیاتی برای research می‌دهد و بدون overhead سرور شروع می‌کند.

---

## مسیر پیشنهادی notebook بعد از این راهنما

| اولویت | Notebook | چرا |
|--------|----------|-----|
| ۱ | `01_us_equities_eda` | شکل پنل و coverage |
| ۲ | `02_corporate_actions` | adjustment واقعی |
| ۳ | `14_point_in_time_validation` | bitemporal / as-of |
| ۴ | `15_survivorship_bias_detection` | اندازه‌گیری bias |
| ۵ | `06_futures_continuous` | roll به‌عنوان تصمیم dataset |
| ۶ | `12_fx_pairs_eda` | convention در OTC |
| ۷ | `20_storage_benchmark_file` | چرا Parquet |
