import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import urllib.parse

# 1. إعداد عنوان وتصميم الصفحة (ستايل راقٍ ومريح للعين)
st.set_page_config(
    page_title="كاشف أورام المخ بالذكاء الاصطناعي",
    page_icon="🧠",
    layout="centered"
)

# تخصيص واجهة المستخدم بـ CSS متطور ليظهر بشكل "فريندلي" ويفتح النفس
st.markdown("""
    <style>
    .main-title {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        text-align: center;
        color: #1E3A8A;
        font-weight: bold;
        margin-top: -20px;
    }
    .sub-title {
        text-align: center;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f5f9;
        color: #334155;
        text-align: center;
        padding: 8px;
        font-size: 13px;
        font-weight: bold;
        border-top: 1px solid #e2e8f0;
        z-index: 100;
    }
    .designer-credit {
        color: #1D4ED8;
        font-family: 'Courier New', Courier, monospace;
    }
    </style>
""", unsafe_allow_html=True)

# 2. تحميل الموديل الذكي (مع ميزة الكاش لتسريع الأداء)
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('brain_tumor_model.h5')
    return model

try:
    model = load_my_model()
except Exception as e:
    st.sidebar.error("⚠️ لم يتم العثور على ملف الموديل 'brain_tumor_model.h5'. يرجى التأكد من وجوده بجانب هذا الملف!")

# 3. قاعدة البيانات الطبية باللغتين
medical_info = {
    'glioma_tumor': {
        'ar': {
            'name': 'الورم الدبقي (Glioma)',
            'desc': 'هو نوع شائع من الأورام التي تنشأ في الخلايا الدبقية التي تدعم وتحمي الخلايا العصبية في الدماغ.',
            'symptoms': 'الصداع المستمر، التشنجات، صعوبة الكلام، أو تغيرات مفاجئة في الرؤية والذاكرة.',
            'advice': 'يتطلب فحصاً عاجلاً مع طبيب مخ وأعصاب لمناقشة خطة العلاج المناسبة (جراحة، إشعاع، أو علاج كيماوي).'
        },
        'en': {
            'name': 'Glioma Tumor',
            'desc': 'A type of tumor that starts in the glial cells, which support and protect the nerve cells in your brain.',
            'symptoms': 'Headaches, seizures, difficulty speaking, or sudden changes in vision and memory.',
            'advice': 'Immediate consultation with a neurologist or neurosurgeon is required to plan treatment (Surgery, Radiation, or Chemotherapy).'
        }
    },
    'meningioma_tumor': {
        'ar': {
            'name': 'الورم السحائي (Meningioma)',
            'desc': 'هو ورم ينشأ من الأغشية التي تغلف الدماغ والنخاع الشوكي (السحايا). في معظم الحالات، يكون هذا الورم حميداً وبطيء النمو.',
            'symptoms': 'صداع تدريجي، ضعف في الذراعين أو الساقين، أو مشاكل في السمع والتوازن.',
            'advice': 'العديد من الأورام السحائية الصغيرة لا تتطلب علاجاً فورياً بل متابعة دورية بالأشعة، أو جراحة بسيطة إذا كان يضغط على الدماغ.'
        },
        'en': {
            'name': 'Meningioma Tumor',
            'desc': 'A tumor that arises from the membranes that surround your brain and spinal cord (meninges). Most are benign and slow-growing.',
            'symptoms': 'Gradual headaches, weakness in arms or legs, or hearing and balance issues.',
            'advice': 'Many small meningiomas don\'t need immediate treatment and can just be monitored, while larger ones may require surgery.'
        }
    },
    'pituitary_tumor': {
        'ar': {
            'name': 'الورم النخامي (Pituitary)',
            'desc': 'هو ورم ينمو في الغدة النخامية الموجودة في قاعدة الدماغ. هذه الغدة مسؤولة عن إفراز العديد من الهرمونات الحيوية في الجسم.',
            'symptoms': 'اضطراب في الرؤية (بسبب الضغط على العصب البصري)، تغيرات هرمونية، إرهاق مستمر، وزيادة أو نقصان الوزن بدون سبب.',
            'advice': 'علاجه غالباً يكون بالأدوية لتنظيم الهرمونات أو بجراحة منظار بسيطة جداً عن طريق الأنف.'
        },
        'en': {
            'name': 'Pituitary Tumor',
            'desc': 'A tumor that grows in the pituitary gland at the base of the brain, which is responsible for regulating most of the body\'s hormones.',
            'symptoms': 'Vision problems (due to pressure on the optic nerve), hormonal changes, fatigue, or unexplained weight shifts.',
            'advice': 'Treatment often involves medication to control hormones, or a minimally invasive endoscopic surgery through the nose.'
        }
    },
    'no_tumor': {
        'ar': {
            'name': 'سليم تماماً (No Tumor)',
            'desc': 'أشعة الرنين المغناطيسي تظهر أن الدماغ سليم تماماً ولا يوجد أي أثر لوجود أورام في المناطق المفحوصة.',
            'symptoms': 'لا توجد أعراض مرتبطة بأورام المخ.',
            'advice': 'حافظ على نمط حياة صحي، وفي حال وجود صداع مستمر يرجى مراجعة الطبيب لفحص مسببات أخرى بسيطة مثل الإرهاق أو الجيوب الأنفية.'
        },
        'en': {
            'name': 'Healthy Brain (No Tumor)',
            'desc': 'The MRI scan shows a perfectly healthy brain structure with no visible signs of tumorous tissues in the examined regions.',
            'symptoms': 'No tumor-related symptoms detected.',
            'advice': 'Maintain a healthy lifestyle. If you still experience minor symptoms like headaches, consult a doctor to check other common causes like stress or sinus issues.'
        }
    }
}

# =========================================================================
# 🛠️ القائمة الجانبية المحدثة (Sidebar): الدخول، الإعدادات، الدعم، والتوثيق
# =========================================================================
st.sidebar.markdown("<h2 style='text-align: center;'>⚙️ لوحة التحكم</h2>", unsafe_allow_html=True)
st.sidebar.write("---")

# 👤 قسم الحساب وتسجيل الدخول
st.sidebar.subheader("👤 الحساب الشخصي / Account")
auth_option = st.sidebar.selectbox("اختر طريقة تسجيل الدخول:", ["زائر (Guest)", "البريد الإلكتروني (Email)", "فيسبوك (Facebook)"])

if auth_option != "زائر (Guest)":
    st.sidebar.text_input("اسم المستخدم أو الإيميل:")
    st.sidebar.text_input("كلمة المرور:", type="password")
    if st.sidebar.button("تسجيل الدخول / Login"):
        st.sidebar.success("🎉 تم تسجيل الدخول بنجاح لمحاكاة النظام!")
else:
    st.sidebar.info("أنت تتصفح كزائر حالياً. يمكنك رفع الأشعة مباشرة.")

st.sidebar.write("---")

# ⚙️ قسم الإعدادات
st.sidebar.subheader("🛠️ الإعدادات / Settings")
theme_mode = st.sidebar.toggle("تفعيل الوضع الليلي (عبر المحاكاة)", value=False)
confidence_threshold = st.sidebar.slider("حد الثقة الأدنى للتصنيف (%)", 50, 95, 70)

st.sidebar.write("---")

# ✉️ قسم الدعم الفني الموجه لإيميل يوسف الشخصي
st.sidebar.subheader("✉️ الدعم الفني / Support")
st.sidebar.write("هل واجهتك مشكلة؟ اكتبها هنا وسيرسل النظام رسالة بريد إلكتروني للمطور مباشرة:")
user_message = st.sidebar.text_area("اكتب مشكلتك أو استفسارك هنا:")

if st.sidebar.button("📧 إرسال الدعم / Send Support"):
    if user_message.strip() != "":
        # تحضير الرابط البريدي بشكل آمن لفتح برنامج الإيميل وتوجيهه ليوسف مباشرة
        subject = urllib.parse.quote("Support Request - Brain Tumor WebApp")
        body = urllib.parse.quote(user_message)
        mailto_url = f"mailto:yosefelosely@gmail.com?subject={subject}&body={body}"
        
        st.sidebar.success("👍 تم تجهيز رسالة الدعم بنجاح!")
        st.sidebar.markdown(f'<a href="{mailto_url}" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #2563EB; color: white; text-align: center; font-weight: bold; border-radius: 5px; text-decoration: none;">انقر هنا لتأكيد الإرسال عبر إيميلك</a>', unsafe_allow_html=True)
    else:
        st.sidebar.warning("الرجاء كتابة المشكلة أولاً قبل الضغط على إرسال.")

st.sidebar.write("---")

# 🏆 توثيق حقوق المصمم المطور في القائمة الجانبية
st.sidebar.markdown("""
    <div style='text-align: center; font-size: 12px; color: #6B7280;'>
        Developed & Designed by <br>
        <span class='designer-credit' style='font-size: 14px;'><b>Youssef Elosely</b></span>
    </div>
""", unsafe_allow_html=True)


# =========================================================================
# 🧠 واجهة التطبيق الأساسية (Main App Body)
# =========================================================================
st.markdown("<h1 class='main-title'>🧠 نظام تشخيص أورام المخ الذكي</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>نظام متطور يعتمد على تقنيات التعلم العميق المتقدمة لمساعدتك في فحص الأشعة فورياً وبشرح تفصيلي للمرض</p>", unsafe_allow_html=True)
st.write("---")

# 🌐 اختيار اللغة لراحة المستخدم وتسهيل التجربة
lang = st.radio(
    "🌐 اختر لغة العرض والتشخيص / Choose Language:",
    ('العربية', 'English'),
    horizontal=True
)

if lang == 'العربية':
    st.subheader("📁 ارفع صورة الأشعة")
    file_label = "اسحب وأسقط صورة أشعة الرنين (MRI) هنا أو اضغط لرفعها"
    text_info = "في انتظار رفع صورة الأشعة للبدء في الفحص والتشخيص الشامل..."
    analyzing_text = "🔄 جاري فحص وتحليل طبقات الأشعة ومطابقتها بالذكاء الاصطناعي..."
    result_header = "📊 التقرير الطبي والتشخيص النهائي للذكاء الاصطناعي:"
    confidence_label = "🎯 نسبة ثقة النموذج في التشخيص:"
    desc_label = "📖 نبذة علمية عن الحالة المكتشفة:"
    symptoms_label = "⚠️ الأعراض الشائعة المصاحبة لها:"
    advice_label = "💡 التوصية الطبية والخطوة التالية:"
else:
    st.subheader("📁 Upload MRI Scan")
    file_label = "Drag and drop your brain MRI image here or click to browse"
    text_info = "Awaiting MRI scan upload to begin the analysis and detailed report..."
    analyzing_text = "🔄 Analyzing and processing MRI scan slices via deep neural network..."
    result_header = "📊 AI Medical Diagnostics & Detailed Report:"
    confidence_label = "🎯 Model Confidence Score:"
    desc_label = "📖 About the Detected Condition:"
    symptoms_label = "⚠️ Common Associated Symptoms:"
    advice_label = "💡 Medical Recommendation & Next Steps:"

# 6. مربع الرفع الذكي للأشعة
file = st.file_uploader(file_label, type=["jpg", "png", "jpeg"])

if file is None:
    st.info(text_info)
else:
    # عرض الصورة بطريقة رشيقة وجميلة تفتح النفس
    image = Image.open(file)
    st.image(image, caption='MRI Scan Mapped Successfully', use_container_width=True)
    
    st.write(analyzing_text)
    
    # تحضير الصورة للموديل بنفس المقاسات والخطوات الرياضية بدقة
    size = (150, 150)
    image_resized = ImageOps.fit(image, size)
    img_array = np.asarray(image_resized)
    
    # معالجة القنوات اللونية والـ Normalization
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
        
    normalized_image_array = (img_array.astype(np.float32) / 255.0)
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # تشغيل التوقع الذكي
    prediction = model.predict(data)
    result_index = np.argmax(prediction)
    confidence_score = prediction[0][result_index] * 100

    # مفاتيح الحالات بالترتيب المتوافق مع الموديل
    keys = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
    detected_key = keys[result_index]

    # جلب تفاصيل التشخيص والشرح بناءً على اللغة المفضلة للمستخدم
    lang_key = 'ar' if lang == 'العربية' else 'en'
    details = medical_info[detected_key][lang_key]

    st.write("---")
    st.subheader(result_header)
    
    # عرض النتيجة بألوان متفاعلة (أخضر لو سليم، أحمر لو ورم)
    if detected_key == 'no_tumor':
        st.success(f"✅ **{details['name']}**")
    else:
        st.error(f"⚠️ **{details['name']}**")
        
    st.info(f"{confidence_label} **{confidence_score:.2f}%**")
    
    # عرض الشرح الطبي المبسط والتوجيهات الروقان!
    st.markdown(f"### {desc_label}")
    st.write(details['desc'])
    
    st.markdown(f"### {symptoms_label}")
    st.write(details['symptoms'])
    
    st.markdown(f"### {advice_label}")
    st.warning(details['advice'])

# =========================================================================
# 🏆 التوقيع النهائي الثابت في أسفل الموقع لتأكيد حقوق التصميم لـ يوسف
# =========================================================================
footer_text = f"🧬 Designed & Built with ❤️ by <span class='designer-credit'><b>Youssef Elosely</b></span> | © 2026 Faculty of Science"
st.markdown(f"<div class='footer'>{footer_text}</div>", unsafe_allow_html=True)