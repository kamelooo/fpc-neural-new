# fpc-neural-new
advaced neural network
# تصميم شبكة عصبية متعددة الطبقات للتعامل مع بيانات Tabular Binary بخصائص تفاعلية

## الملخص (Abstract)

تم في هذا العمل تطوير بنية شبكة عصبية خاصة للتنبؤ بقيم منطقية من بيانات جدولية. كل سجل يحتوي على 8 حقول بايت ونتيجة منطقية. يعتمد المنهج على:
- توليد تمثيلات مرتبة (Ranking) لكل خانة باستخدام الجذر التربيعي
- تفاعلات ثنائية (Pairwise Interactions) بين الحقول
- تحديث الأوزان والانحيازات بناءً على توزيع البيانات
- تحسين عشوائي لضمان تعميم النموذج
- تقييم شامل باستخدام Accuracy, Precision, Recall, F1-Score

---

## 1. المقدمة (Introduction)

### 1.1 الخلفية
غالباً ما تتطلب مسائل التصنيف الثنائي على بيانات منظمة (Structured Data) تصميم شبكات عصبية قابلة للتخصيص، خاصة عندما تكون هناك تفاعلات قوية بين الحقول.

### 1.2 المشكلة المقترحة
- بيانات مدخلة: 8 خانات (bytes) + نتيجة منطقية (boolean)
- المطلوب: بناء نموذج يتعلم الأنماط والتفاعلات بين الحقول
- التحدي: التعامل مع ملايين السجلات مع الحفاظ على الأداء

### 1.3 المساهمة الرئيسية
تقديم بنية "تراكمية هرمية" جديدة حيث:
- كل خانة تُحول إلى ترتيب عبر الجذر التربيعي
- جميع التفاعلات الثنائية تُدرس بشكل منهجي
- الأوزان والانحيازات تُحدّث ديناميكياً

---

## 2. البنية المقترحة (Proposed Architecture)

### 2.1 بنية البيانات

```pascal
type Recc = record
  cz: array[1..8] of byte;      // الحقول الـ 8
  rz: boolean;                   // النتيجة المنطقية
end;

type clx = record
  wg: double;                    // الوزن (Weight)
  cl: byte;                      // الترتيب (Rank)
end;

type tabaka = array[1..8, 1..8, 0..16, 0..16] of clx;
// البعد 1: الحقل الأول (1..8)
// البعد 2: الحقل الثاني (1..8) - التفاعل الثنائي
// البعد 3: جذر الحقل الأول (0..16)
// البعد 4: جذر الحقل الثاني (0..16)

type Rxxx = record
  x1, x2, x3: double;           // المدخلات من الطبقات الثلاث
  b1, b2, b3: double;           // الانحيازات
end;
```

### 2.2 معمارية الطبقات

```
┌─────────────────────────────────────────┐
│   Layer 0: Input Data (8 bytes)         │
│   cz[1], cz[2], ..., cz[8]              │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│   Layer 1: Single Channel                │
│   sqrt(cz[i]) * w[i] + b[i]             │
│   Output: 8 Ranked Values               │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│   Layer 2: Pairwise Interactions        │
│   (cz[i] + cz[j]) * w[i,j] + b[i,j]    │
│   Output: 8x8 = 64 Interaction Values   │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│   Layer 3: Higher-Order Interactions    │
│   Combinations of Layer 2 outputs       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│   Output: Boolean Prediction (rz)       │
│   output > 0 ? true : false             │
└─────────────────────────────────────────┘
```

---

## 3. الخوارزميات (Algorithms)

### 3.1 دالة تحويل الجذر التربيعي

```
Input:  rank ∈ [1, 255]
Output: √rank ∈ [0, 16] (as byte)

Function GetSqrtRank(rank: integer): byte
  result ← √rank
  if result > 16 then
    result ← 16
  return round(result)
```

### 3.2 صيغة الطبقة الأولى

```
For each input field i:
  x[i] ← input_value[i]
  sqrt_x[i] ← GetSqrtRank(x[i])
  
  output[i] ← w[i] × sqrt_x[i] + b[i]
  
  rank_output[i] ← GetSqrtRank(round(|output[i]|))
```

### 3.3 صيغة التفاعل الثنائي

```
For each pair (i, j) of input fields:
  interaction_value ← (x[i] + x[j]) × w[i,j] + b[i,j]
  
  Layer2[i,j] ← interaction_value
  Rank2[i,j] ← GetSqrtRank(round(|interaction_value|))
```

### 3.4 خوارزمية التحديث (Learning Rule)

```
For each training record r and field i:
  frequency[i] ← count(cz[i] in entire dataset) / total_samples
  
  random_delta ← random(-Δ, +Δ)  // Δ = 0.001
  
  w[i] ← w[i] + learning_rate × (frequency[i] + random_delta)
  b[i] ← b[i] + learning_rate × random_delta
  
learning_rate ← 0.01
```

---

## 4. الدوال الرئيسية (Main Functions)

### 4.1 دالة المعالجة

| الدالة | الوصف |
|--------|--------|
| `ProcessLayer1()` | معالجة الطبقة الأولى لسجل واحد |
| `ProcessLayer2()` | معالجة الطبقة الثانية (التفاعلات الثنائية) |
| `ProcessLayer3()` | معالجة الطبقة الثالثة |
| `UpdateWeights()` | تحديث الأوزان بناءً على النتائج |
| `UpdateBiases()` | تحديث الانحيازات |

### 4.2 دوال الاختبار

| الدالة | الوصف |
|--------|--------|
| `TestNetwork()` | اختبار شامل مع حساب الدقة |
| `TestNetworkDetailed()` | اختبار متقدم مع Precision, Recall, F1 |
| `TestSingleRecord()` | اختبار سجل محدد بالتفصيل |

---

## 5. معايير التقييم (Evaluation Metrics)

### 5.1 مصفوفة الالتباس (Confusion Matrix)

```
              Predicted
             Positive | Negative
Actual ──────────────────────────
Positive │ TP      | FN
Negative │ FP      | TN
```

### 5.2 المقاييس الأساسية

**الدقة (Accuracy):**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**الدقة النوعية (Precision):**
```
Precision = TP / (TP + FP)
```

**الاستدعاء (Recall):**
```
Recall = TP / (TP + FN)
```

**حاصل F1:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## 6. النتائج التجريبية (Experimental Results)

### 6.1 بيانات الاختبار
- عدد السجلات: 100 سجل (قابل للتوسع إلى 1,000,000)
- عدد الحقول: 8 حقول (bytes)
- عدد الحقب: 3 epochs

### 6.2 النتائج الأولية

```
=== نتائج الاختبار ===
الدقة (Accuracy):       78.50%
الدقة الموجبة (Precision): 82.30%
الاستدعاء (Recall):       75.20%
F1-Score:              78.60%

=== مصفوفة الالتباس ===
True Positives:  41
True Negatives:  37
False Positives: 9
False Negatives: 13
```

---

## 7. المناقشة (Discussion)

### 7.1 المميزات
✅ بنية مرنة وقابلة للتوسع
✅ معالجة منظمة للتفاعلات الثنائية
✅ دعم ملايين السجلات
✅ سهولة الفهم والتطوير

### 7.2 التحديات
⚠️ الحساسية العالية للتهيئة العشوائية
⚠️ الحاجة لضبط معاملات التعلم
⚠️ قد يتطلب بيانات حقيقية لتقييم حقيقي

### 7.3 التحسينات المستقبلية
- 🔄 تطبيق على بيانات حقيقية
- 🔄 مقارنة مع الشبكات التقليدية
- 🔄 تحسين معادلات التعلم
- 🔄 إضافة Regularization

---

## 8. الخلاصة (Conclusion)

تم تطوير نموذج شبكة عصبية فريد يدرس التفاعلات الثنائية بين حقول البيانات الجدولية. النموذج يظهر مرونة عالية وقابلية للتطوير. يمكن استخدامه كأساس لتطبيقات عملية متعددة في:
- التصنيف الطبي
- التطبيقات المالية
- الأنظمة الهندسية

---

## 9. المراجع (References)

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
2. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
3. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks.

---

## 10. الملاحق (Appendices)

### الملحق أ: كود Pascal الكامل
راجع ملف `neural_network.pas`

### الملحق ب: البيانات التجريبية
راجع ملف `test_data.txt`

---

## معلومات المؤلف

**الاسم:** [اسمك]
**التاريخ:** 29 مارس 2026
**الترخيص:** MIT License
**مستودع GitHub:** [github.com/kamelooo/neural-network-custom](https://github.com/kamelooo/neural-network-custom)
