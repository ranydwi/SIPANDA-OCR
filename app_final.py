# Bismillah Aktualisasi 
# UI enhanced: DJP header + sidebar logo (Streamlit Prophet) + theme colors
import streamlit as st
from io import BytesIO
import zipfile
import re
import os
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# PDF / OCR libs
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageOps, ImageFilter

# import base
import base64

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="SIPANDA-OCR",
    layout="centered"
)

# =========================
# LOGIN CONFIG
VALID_USERS = {
    "admin": "admin453",
    "pegawai": "pajak453"
}

# =========================
# SESSION STATE INIT

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# =========================
# Uncomment & set if Tesseract not in PATH (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OCR_DPI = 200            # render once at moderate dpi
HEADER_CROP = (0.02, 0.02, 0.98, 0.30)
NUMBER_CROP = (0.05, 0.20, 0.95, 0.55)
MAX_WORKERS = max(2, min(8, (multiprocessing.cpu_count() or 4)))

# Theme / logos — replace URL with direct image link or use local file paths
# IMPORTANT: Use direct image URLs (ending with .png/.jpg) or put local path (e.g., "assets/djp_logo.png")
LOGO_URL = "img\logo_kpp.PNG"        # <- put DJP logo direct link here

# Theme colors
COLOR_BLUE = "#002b5c"
COLOR_YELLOW = "#FFD200"

# Only strict billing keywords here; visual & currency checks used too
BILLING_KEYWORDS = ["billing", "invoice"]
FALLBACK_NAME = "UNKNOWN"

DOC_TYPES = {
    "SKP_PPN": {"keywords": ["pajak pertambahan nilai", "ppn"], "nomor_regex": r"([0-9]{3,6}(?:/[0-9]{1,4}){3,})", "chunk_size": 4},
    "STP":     {"keywords": ["surat tagihan pajak", "tagihan pajak", "surat tagihan", "tagihan"], "nomor_regex": r"([0-9]{3,6}(?:/[0-9]{1,4}){3,})", "chunk_size": 2},
    "SKPKB":   {"keywords": ["kurang bayar"], "nomor_regex": r"Nomor[:\s]*([0-9A-Za-z\/\-\.\s]+)", "chunk_size": 2},
    "SKPLB":   {"keywords": ["lebih bayar"], "nomor_regex": r"Nomor[:\s]*([0-9A-Za-z\/\-\.\s]+)", "chunk_size": 9999},
    "SKPN":    {"keywords": ["nihil"], "nomor_regex": r"Nomor[:\s]*([0-9A-Za-z\/\-\.\s]+)", "chunk_size": 2}
}

# ---------------- UTILITIES ----------------
# =========================
# LOGIN PAGE
# =========================
def login_page():
    #logo_base64_left = img_to_base64("img\logo_djp_wh.png")
    logo_base64_right = img_to_base64("img\logoponaren.png")
    logo_base64_rp = img_to_base64(r"img\rp_white.png")
    st.markdown("""
    <style>
    /* ===== FORCE LOGIN BACKGROUND ===== */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(
            180deg,
            #002b5c 0%,
            #001f42 100%
        ) !important;
    }

    [data-testid="stMain"] {
        background: transparent !important;
    }

    .block-container {
        padding-top: 6rem;
        max-width: 520px;
    }

    div[data-testid="stForm"] {
        background: #ffffff;
        padding: 28px 30px;
        border-radius: 14px;
        box-shadow: 0 14px 35px rgba(0,0,0,0.25);
    }

    h1, h2, h3, p {
        color: white !important;
    }

    div[data-testid="stForm"] label,
    div[data-testid="stForm"] p,
    div[data-testid="stForm"] span {
        color: #002b5c !important;
        font-weight: 600;
    }

    button[kind="primary"] {
        width: 100%;
        background-color: #FFD200 !important;
        color: #002b5c !important;
        font-weight: 800;
        border-radius: 8px;
        padding: 10px;
    }
                
    /* ===== LOGO FIX POJOK KIRI ATAS ===== */
    .login-logo-fixed-left {
        position: fixed;
        top: 16px;
        left: 18px;
        z-index: 99999;

        background: #002b5c;
        padding: 8px 12px;
        border-radius: 10px;

        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    }

    .login-logo-fixed-left img {
        height: 56px;
        width: auto;
        display: block;
    }
                
    # .login-logo-fixed-right {
    #     position: fixed;
    #     top: 16px;
    #     left: 1240px;
    #     z-index: 99999;

    #     background: #002b5c;
    #     padding: 8px 12px;
    #     border-radius: 10px;

    #     box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    # }

    # .login-logo-fixed-right img {
    #     height: 40px;
    #     width: auto;
    #     display: block;
    # }

    /* ===== LOGIN FOOTER BAR (DJP STYLE) ===== */
    .login-footer-bar {
        position: fixed;
        bottom: 16px;
        left: 24px;
        right: 24px;

        display: flex;
        align-items: center;
        gap: 18px;

        z-index: 9999;
    }

    /* Garis panjang */
    .login-footer-line {
        flex: 1;
        height: 5px;
        display: flex;
        border-radius: 2px;
        overflow: hidden;
    }

    /* Segmen warna */
    .login-footer-line .seg-white {
        flex: 1;
        background: #ffffff;
    }

    .login-footer-line .seg-yellow {
        flex: 1;
        background: #FFD200;
    }

    .login-footer-line .seg-yellow-light {
        flex: 1;
        background: #FFEB3B;
    }

    /* Teks pajak.go.id */
    .login-footer-text {
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 0.8px;
        color: #ffffff;
        white-space: nowrap;
    }

    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="login-logo-fixed-left">
            <img src="data:image/png;base64,{logo_base64_right}" alt="Logo DJP">
        </div>
        """,
        unsafe_allow_html=True
    )

    # st.markdown(
    #     f"""
    #     <div class="login-logo-fixed-right">
    #         <img src="data:image/png;base64,{logo_base64_rp}" alt="Logo RP">
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )

    st.title("🔐 Login SIPANDA-OCR")
    st.caption(
        "Sistem Pemisahan dan Penamaan Dokumen berbasis OCR "
        "untuk STP, SKPKB, SKPLB, dan SKPN"
    )

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if username in VALID_USERS and VALID_USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login berhasil")
                st.rerun()
            else:
                st.error("Username atau password salah")

    # 👇 FOOTER PALING BAWAH
    st.markdown("""
    <div class="login-footer-bar">
        <div class="login-footer-line">
            <div class="seg-white"></div>
            <div class="seg-yellow"></div>
            <div class="seg-yellow-light"></div>
        </div>
        <div class="login-footer-text">www.pajak.go.id</div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------
def render_page_image_bytes(file_bytes: bytes, page_index: int, dpi=OCR_DPI):
    try:
        imgs = convert_from_bytes(file_bytes, dpi=dpi, first_page=page_index+1, last_page=page_index+1)
        return imgs[0] if imgs else None
    except Exception:
        return None

def render_all_pages(file_bytes: bytes, total: int, dpi=OCR_DPI, max_workers=MAX_WORKERS):
    imgs = [None] * total
    def job(i):
        try:
            pages = convert_from_bytes(file_bytes, dpi=dpi, first_page=i+1, last_page=i+1)
            return i, pages[0] if pages else None
        except Exception:
            return i, None
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(job, i) for i in range(total)]
        for fut in as_completed(futures):
            i, img = fut.result()
            imgs[i] = img
    return imgs

def crop(img: Image.Image, frac: Tuple[float,float,float,float]) -> Image.Image:
    w,h = img.size
    l,t,r,b = frac
    return img.crop((int(w*l), int(h*t), int(w*r), int(h*b)))

def ocr_image(img: Image.Image, config: str = "--psm 6", lang: str = "ind+eng"):
    try:
        return pytesseract.image_to_string(img, lang=lang, config=config)
    except Exception:
        return pytesseract.image_to_string(img)

# ---------------- billing detection ----------------
def is_billing_page(image: Image.Image, full_text: str, debug: bool=False) -> bool:
    txt = (full_text or "").lower()

    # 1) direct keywords in full text
    if re.search(r'\bbilling\b', txt) or re.search(r'\binvoice\b', txt):
        if debug:
            st.write("Billing detected by keyword in full_text")
        return True

    # 2) currency / total pattern
    if (("jumlah" in txt and ("rp" in txt or "rupiah" in txt)) or
        ("total" in txt and re.search(r'\brp\b|\brupiah\b|\d{3,}', txt))):
        if debug:
            st.write("Billing detected by currency/total pattern in full_text")
        return True

    # 3) visual crop check (center-top area where BILLING often large)
    try:
        w,h = image.size
        crop_box = (int(w*0.10), int(h*0.05), int(w*0.90), int(h*0.35))
        crop_img = image.crop(crop_box)
        up = crop_img.resize((crop_img.width*2, crop_img.height*2), Image.LANCZOS)
        up = ImageOps.autocontrast(up)
        up = up.filter(ImageFilter.SHARPEN)
        crop_txt = ocr_image(up, config="--psm 6").lower()
        if debug:
            st.text("Crop OCR (billing test) -> " + crop_txt[:200])
        if "billing" in crop_txt or "invoice" in crop_txt or "bill" in crop_txt:
            if debug:
                st.write("Billing detected by visual crop OCR")
            return True
    except Exception: 
        pass

    return False

# ---------------- classifier & extraction ----------------
def classify(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in DOC_TYPES["SKP_PPN"]["keywords"]):
        return "SKP_PPN"
    for k,cfg in DOC_TYPES.items():
        if k == "SKP_PPN": continue
        if any(w in t for w in cfg["keywords"]):
            return k
    return "UNKNOWN"

def robust_extract_nomor(file_bytes: bytes, pages: List[int], images: List[Optional[Image.Image]], headers: List[str], debug: bool=False) -> Optional[str]:
    candidate = ""
    first_idx = pages[0] if pages else 0
    if 0 <= first_idx < len(headers):
        candidate += (headers[first_idx] or "") + "\n"

    whitelist_cfg = r'--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789OIl\/\-\.\s'
    for p in pages:
        img = None
        if images and p < len(images):
            img = images[p]
        if img is None:
            img = render_page_image_bytes(file_bytes, p, dpi=OCR_DPI)
        if img is None:
            continue
        c = crop(img, NUMBER_CROP)
        try:
            gray = c.convert("L")
            gray = ImageOps.autocontrast(gray)
            gray = gray.filter(ImageFilter.SHARPEN)
            gray = gray.point(lambda x: 0 if x < 110 else 255)
        except Exception:
            gray = c
        try:
            txt = pytesseract.image_to_string(gray, lang="ind+eng", config=whitelist_cfg)
        except Exception:
            txt = pytesseract.image_to_string(gray)
        if txt and len(txt.strip()) > 1:
            candidate += txt + "\n"

    if len(candidate.strip()) < 4:
        img = None
        if images and first_idx < len(images):
            img = images[first_idx]
        if img is None:
            img = render_page_image_bytes(file_bytes, first_idx)
        if img is not None:
            big = crop(img, (0.01,0.01,0.99,0.70))
            try:
                txt = pytesseract.image_to_string(big, lang="ind+eng", config="--psm 6 --oem 3")
            except Exception:
                txt = pytesseract.image_to_string(big)
            candidate += txt or ""

    if debug:
        st.text("---- kandidat OCR (raw) ----")
        st.code(candidate[:2000])

    if not candidate.strip():
        if debug:
            st.error("No OCR candidate text.")
        return None

    cand = candidate.upper()
    cand = cand.replace("：", ":").replace("–", "-").replace("—", "-")
    cand = cand.replace("|", "/")
    cand = cand.replace("O", "0").replace("I", "1").replace("L", "1")
    cand = re.sub(r"[^\d\/\.\-\s:]", " ", cand)
    cand = re.sub(r"\s+", " ", cand).strip()

    if debug:
        st.text("---- kandidat OCR (normalized) ----")
        st.code(cand[:2000])

    regexes = [
        r'([0-9]{3,6}[\s\/\.\-][0-9]{1,4}[\s\/\.\-][0-9]{1,4}[\s\/\.\-][0-9]{1,4}[\s\/\.\-][0-9]{1,4})',
        r'([0-9]{3,6}[\s\/\.\-][0-9]{1,4}[\s\/\.\-][0-9]{1,4}[\s\/\.\-][0-9]{1,4})',
        r'Nomor[:\s]*([0-9\.\-\/\s]+)',
        r'NO\.?[:\s]*([0-9\.\-\/\s]+)'
    ]

    for rg in regexes:
        m = re.search(rg, cand, flags=re.IGNORECASE)
        if m:
            raw = m.group(1)
            raw = re.sub(r"[ \t]+", "", raw)
            raw = raw.replace("/", ".").replace("-", ".")
            raw = re.sub(r"\.+", ".", raw)
            raw = raw.strip(".")
            if debug:
                st.success(f"Regex match: {raw}")
            return raw

    loose = re.search(r'([0-9][0-9\/\.\-\s]{8,40}[0-9])', cand)
    if loose:
        raw = loose.group(1)
        raw = re.sub(r"[ \t]+", "", raw)
        raw = raw.replace("/", ".").replace("-", ".")
        raw = re.sub(r"\.+", ".", raw).strip(".")
        if debug:
            st.warning(f"Loose-match fallback: {raw}")
        return raw

    if debug:
        st.error("No nomor match found.")
    return None

def extract_nama_wajib_pajak(
    file_bytes: bytes,
    pages: List[int],
    images: List,
    headers: List[str],
    debug: bool = False
) -> Optional[str]:

    text_blob = ""

    first_idx = pages[0] if pages else 0

    # =========================
    # HEADER OCR
    # =========================
    if 0 <= first_idx < len(headers):
        text_blob += (headers[first_idx] or "") + "\n"

    # =========================
    # BODY OCR (AREA NAMA)
    # =========================
    img = images[first_idx] if images and first_idx < len(images) else None
    if img is not None:
        # area tengah-atas (aman untuk STP)
        big = crop(img, (0.05, 0.25, 0.95, 0.55))
        try:
            txt = pytesseract.image_to_string(
                big, lang="ind+eng", config="--psm 6"
            )
        except Exception:
            txt = ""
        text_blob += "\n" + txt

    if debug:
        st.text("---- OCR AREA NAMA WP ----")
        st.code(text_blob[:1200])

    if not text_blob.strip():
        return None

    cand = (
        text_blob
        .replace("：", ":")
        .replace(">", ":")
        .replace("=", ":")
    )

    # =========================
    # REGEX UTAMA (KERAS)
    # =========================
    patterns = [
        r'Nama\s+Wajib\s+Pajak\s*/\s*PKP\s*[:]\s*(.+)',
        r'Nama\s+Wajib\s+Pajak\s*[:]\s*(.+)',
        r'Nama\s+Wajib\s+Pajak\s*/\s*PKP\s*(.+)',
    ]

    for rg in patterns:
        m = re.search(rg, cand, flags=re.IGNORECASE)
        if m:
            val = m.group(1).split("\n")[0].strip(" .:-")

            # HARD FILTER KOP
            if re.search(
                r'kementerian|direktorat|kantor pelayanan',
                val,
                flags=re.IGNORECASE
            ):
                continue

            val = re.sub(r'\s{2,}', ' ', val)

            if debug:
                st.success(f"Nama WP (regex): {val}")

            return val

    # =========================
    # FALLBACK CERDAS (ANTI KOP)
    # =========================
    blacklist = [
        "kementerian",
        "direktorat",
        "kantor pelayanan",
        "surat",
        "pajak pertambahan",
        "surat tagihan",
        "tanggal",
        "masa pajak",
        "jumlah rupiah"
    ]

    lines = [ln.strip() for ln in cand.splitlines() if ln.strip()]

    for ln in lines:
        lo = ln.lower()

        if any(b in lo for b in blacklist):
            continue

        # ciri nama WP
        if (
            re.search(r'\bPT\b|\bCV\b|\bUD\b', ln)
            or ln.isupper()
        ):
            if 2 <= len(ln.split()) <= 8:
                if debug:
                    st.warning(f"Nama WP (fallback aman): {ln}")
                return ln.strip()

    if debug:
        st.error("Nama WP tidak ditemukan (final).")

    return None

def pages_to_pdf(reader: PdfReader, page_indices: List[int]) -> bytes:
    writer = PdfWriter()
    for idx in page_indices:
        writer.add_page(reader.pages[idx])
    out = BytesIO()
    writer.write(out)
    return out.getvalue()

# ---------------- core processing (same as your working code) ----------------
def process_for_type(file_bytes: bytes, filename: str, target: str, debug: bool=False) -> List[Tuple[str, bytes]]:
    results: List[Tuple[str, bytes]] = []
    reader = PdfReader(BytesIO(file_bytes))
    total_pages = len(reader.pages)
    if total_pages == 0:
        return results

    # render all pages once (parallel)
    images = render_all_pages(file_bytes, total_pages, dpi=OCR_DPI, max_workers=MAX_WORKERS)

    # precompute header-area OCR (fast) -- hATI-
    def ocr_header_img(img):
        if img is None: return ""
        try:
            h_crop = crop(img, HEADER_CROP)
            return ocr_image(h_crop, config="--psm 6") or ""
        except Exception:
            return ""
    headers = [""] * total_pages
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(ocr_header_img, images[i]): i for i in range(total_pages)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                headers[i] = fut.result() or ""
            except Exception:
                headers[i] = ""

    if debug:
        st.write(f"Header indices (preview): {[i for i,h in enumerate(headers) if h.strip()][:30]}")
        for i,h in enumerate(headers[:min(12, total_pages)]):
            st.text(f"PAGE {i+1:02d} headerOCR -> { (h[:200] + '...') if len(h)>200 else h }")

    # detect header indices conservatively
    header_indices = []
    for i,h in enumerate(headers):
        lo = (h or "").lower()
        if ("surat" in lo and "pajak" in lo) or "kementerian keuangan" in lo or "direktorat jenderal pajak" in lo:
            header_indices.append(i)

    doc_ranges: List[Tuple[int,int]] = []

    cached_nama_wp = None

    # STP: forced sequential 2 pages + billing (with safer billing check)
    if target == "STP":
        doc_ranges = []
        i = 0
        while i < total_pages:
            start = i
            end = min(i + 1, total_pages - 1)  # default chunk = 2 pages (0-based)

            nxt = end + 1
            if nxt < total_pages:
                next_header = (headers[nxt] or "").lower()
                # EXCLUDE pages that are clearly LAMPIRAN or just kop (very likely NOT billing)
                if "lampiran" in next_header or ("surat" in next_header and "tagihan" in next_header and "lampiran" in next_header):
                    attach = False
                else:
                    # quick light checks from header-area OCR (cheap)
                    next_light = next_header

                    # 1) direct keyword (strong)
                    kw_flag = bool(re.search(r'\bbilling\b|\binvoice\b', next_light))

                    # 2) currency/amount hint but require stronger evidence (not only "jumlah" in kop)
                    currency_hint = ("jumlah set" in next_light) or ("jumlah setor" in next_light) \
                                    or (("jumlah" in next_light) and ("rp" in next_light or "rupiah" in next_light)) \
                                    or bool(re.search(r'\brp[\.\s]?\d', next_light))

                    # 3) visual crop + full-page OCR only when currency_hint OR ambiguous
                    visual_flag = False
                    if currency_hint or not next_light.strip():
                        # perform visual crop check (center-top + big-area) for "BILLING" or "JUMLAH SETOR" or large numeric block
                        try:
                            img = images[nxt] if images and nxt < len(images) else None
                            if img is not None:
                                # crop center-top to detect big "BILLING" word
                                w,h = img.size
                                crop_box = (int(w*0.10), int(h*0.03), int(w*0.90), int(h*0.35))
                                crop_img = img.crop(crop_box).resize((int(w*0.8*2), int(h*0.32*2)), Image.LANCZOS)
                                crop_img = ImageOps.autocontrast(crop_img)
                                crop_txt = ocr_image(crop_img, config="--psm 6").lower()
                                # debug info printed below if debug True
                                # strong visual evidence if 'billing' or 'jumlah setor' or 'no. sk' or 'no sk' present
                                if re.search(r'\bbilling\b|\binvoice\b', crop_txt) or "jumlah setor" in crop_txt or re.search(r'no\.?\s*sk', crop_txt):
                                    visual_flag = True
                                # also detect big numeric block (billing code) like 15+ digits in center
                                if re.search(r'\d{10,}', crop_txt):
                                    visual_flag = True
                        except Exception:
                            visual_flag = False

                    # final attach decision: require either kw_flag OR visual_flag with currency_hint
                    attach = False
                    if kw_flag:
                        attach = True
                    elif visual_flag and currency_hint:
                        attach = True
                    else:
                        attach = False

                if attach:
                    end = nxt

            doc_ranges.append((start, end))
            i = end + 1


    # SKPKB: group by header blocks
    elif target == "SKPKB":
        doc_ranges = []

        def looks_like_header(text: str) -> bool:
            if not text: return False
            t = text.lower()
            # require kop + 'surat ketetapan' or explicit 'kurang bayar'
            return ("kementerian keuangan" in t and "direktorat jenderal pajak" in t) and (
                ("surat ketetapan" in t) or ("kurang bayar" in t) or ("surat ketetapan pajak" in t)
            )

        def is_ppn_text(text: str) -> bool:
            t = (text or "").lower()
            return ("pajak pertambahan nilai" in t) or (" ppn " in f" {t} ") or ("pertambahan nilai" in t)

        def is_pph_text(text: str) -> bool:
            t = (text or "").lower()
            return ("pajak penghasilan" in t) or (" pph " in f" {t} ") or ("penghasilan" in t)

        def ocr_first_page_large(file_bytes: bytes, page_idx: int) -> str:
            # render and OCR a larger header+title area for better classification
            try:
                img = images[page_idx] if images and page_idx < len(images) else render_page_image_bytes(file_bytes, page_idx, dpi=OCR_DPI)
                if img is None:
                    return ""
                big = img.crop((0, 0, img.width, int(img.height * 0.45)))
                try:
                    big = ImageOps.autocontrast(big)
                except Exception:
                    pass
                return ocr_image(big, config="--psm 6") or ""
            except Exception:
                return ""

        # Find strong header starts
        strong_headers = []
        for i, h in enumerate(headers):
            if looks_like_header(h):
                strong_headers.append(i)

        # If none detected using headers (rare), try scanning every page header OCR to find kop+surat
        if not strong_headers:
            for i, h in enumerate(headers):
                # reuse conservative check but also test header OCR for 'surat ketetapan' substring
                if h and ("surat ketetapan" in h.lower() or "kurang bayar" in h.lower()):
                    strong_headers.append(i)

        # If still none, fallback: try OCR full first-page text for each page and find kop
        if not strong_headers:
            for i in range(total_pages):
                txt = ocr_first_page_large(file_bytes, i)
                if "kementerian keuangan" in txt.lower() and "surat ketetapan" in txt.lower():
                    strong_headers.append(i)

        # sort and unique
        strong_headers = sorted(list(dict.fromkeys(strong_headers)))

        if debug:
            st.write(f"SKPKB: detected strong headers -> {strong_headers}")

        # Build ranges per header (FIXED)
        for idx_pos, start in enumerate(strong_headers):
            hdr_text = headers[start] or ""

            if is_ppn_text(hdr_text):
                is_ppn = True
            elif is_pph_text(hdr_text):
                is_ppn = False
            else:
                fallback_txt = ocr_first_page_large(file_bytes, start)
                is_ppn = is_ppn_text(fallback_txt)

            size = 4 if is_ppn else 2

            if idx_pos + 1 < len(strong_headers):
                limit_end = strong_headers[idx_pos + 1] - 1
            else:
                limit_end = total_pages - 1

            desired_end = min(start + size - 1, total_pages - 1)
            end = min(desired_end, limit_end)

            doc_ranges.append((start, end))

        # If no headers found at all, fallback to sequential chunking (size=2 default; detect PPN via OCR at each start)
        if not doc_ranges:
            i = 0
            while i < total_pages:
                hdr = headers[i] or ""
                is_ppn = is_ppn_text(hdr)
                if not (is_ppn or is_pph_text(hdr)):
                    full = ocr_first_page_large(file_bytes, i)
                    is_ppn = is_ppn_text(full)
                    if debug:
                        st.text(f"SKPKB fallback sequential OCR page {i+1} -> {full[:150]}")
                size = 4 if is_ppn else 2
                start = i
                end = min(start + size - 1, total_pages - 1)
                doc_ranges.append((start, end))
                i = end + 1

    # SKPN ------------------------------
    elif target == "SKPN":
        doc_ranges = []

        def ocr_first_page_large(file_bytes: bytes, page_idx: int) -> str:
            try:
                img = images[page_idx] if images and page_idx < len(images) else render_page_image_bytes(file_bytes, page_idx, dpi=OCR_DPI)
                if img is None:
                    return ""
                big = img.crop((0, 0, img.width, int(img.height * 0.45)))
                return ocr_image(big, config="--psm 6") or ""
            except Exception:
                return ""

        # =============================
        # 1️⃣ DETEKSI HALAMAN AWAL SKPN
        # =============================
        strong_headers = []

        for i, h in enumerate(headers):
            if not h:
                continue
            t = h.lower()

            if (
                "kementerian keuangan" in t and
                "direktorat jenderal pajak" in t and
                "surat ketetapan pajak nihil" in t and
                not "lampiran" in t
            ):
                strong_headers.append(i)

        strong_headers = sorted(set(strong_headers))

        if debug:
            st.write("SKPN strong headers:", strong_headers)

        # ==========================================
        # 2️⃣ BUILD RANGE TANPA idx_pos (AMAN TOTAL)
        # ==========================================
        for i in range(len(strong_headers)):
            start = strong_headers[i]

            # batas akhir = sebelum header berikutnya
            if i + 1 < len(strong_headers):
                limit_end = strong_headers[i + 1] - 1
            else:
                limit_end = total_pages - 1

            # SKPN PPN = 4 halaman, default = 2
            hdr_text = headers[start] or ""
            is_ppn = "ppn" in hdr_text.lower() or "pertambahan nilai" in hdr_text.lower()

            size = 4 if is_ppn else 2
            desired_end = min(start + size - 1, total_pages - 1)

            end = min(desired_end, limit_end)

            doc_ranges.append((start, end))

        # ==================================
        # 3️⃣ FALLBACK SEQUENTIAL (JIKA GAGAL)
        # ==================================
        if not doc_ranges:
            i = 0
            while i < total_pages:
                hdr = headers[i] or ""
                is_ppn = "ppn" in hdr.lower() or "pertambahan nilai" in hdr.lower()
                size = 4 if is_ppn else 2

                start = i
                end = min(i + size - 1, total_pages - 1)

                doc_ranges.append((start, end))
                i = end + 1

   # merge adjacent ranges ONLY for SKPLB (other types should remain separated)
    # SKPLB ------------------------------
    elif target == "SKPLB":
        doc_ranges = []

        def looks_like_skplb_header(text: str) -> bool:
            if not text:
                return False
            t = text.lower()
            if not ("kementerian keuangan" in t and "direktorat jenderal pajak" in t):
                return False
            if "surat ketetapan pajak lebih bayar" not in t:
                return False
            if "lampiran skplb" in t:
                return False
            return True

        strong_headers = []
        for i, h in enumerate(headers):
            if looks_like_skplb_header(h):
                strong_headers.append(i)

        strong_headers = sorted(set(strong_headers))

        if debug:
            st.write("SKPLB headers:", strong_headers)

        for i in range(len(strong_headers)):
            start = strong_headers[i]
            end = strong_headers[i + 1] - 1 if i + 1 < len(strong_headers) else total_pages - 1
            doc_ranges.append((start, end))

        if not doc_ranges:
            doc_ranges = [(0, total_pages - 1)]

# SKPKB / SKPN: header-based preferred, fallback sequential -------------------------
    else:
        if target in ("SKPN", "SKPKB"):
            pass 
        else:
            if header_indices:
                for j, start in enumerate(header_indices):
                    hdr = (headers[start] or "").lower()
                    doc_key = classify(hdr)

                    if doc_key in ("SKPKB", "SKPN") and ("ppn" in hdr or "pertambahan nilai" in hdr):
                        size = 4
                    else:
                        size = DOC_TYPES.get(doc_key, {}).get("chunk_size", 2)

                    desired_end = min(start + size - 1, total_pages - 1)
                    limit_end = header_indices[j+1] - 1 if j+1 < len(header_indices) else total_pages - 1
                    end = min(desired_end, limit_end)
                    doc_ranges.append((start, end))

            if not doc_ranges:
                i = 0
                while i < total_pages:
                    size = DOC_TYPES.get(target, {}).get("chunk_size", 2)
                    start = i
                    end = min(start + size - 1, total_pages - 1)
                    doc_ranges.append((start, end))
                    i = end + 1

    # sanitize nama for filename ----------------------------------------------
    for (start, end) in doc_ranges:
        pages = list(range(start, end + 1))
        out_bytes = pages_to_pdf(reader, pages)

        nomor = robust_extract_nomor(
            file_bytes, pages, images, headers, debug=debug
        )

        # =====================================
        # NAMA WP: LOGIKA BENAR SESUAI JENIS
        # =====================================
        if target == "STP":
            # ✅ STP: ambil nama PER CHUNK
            # (chunk sudah 2–3 halaman, billing aman)
            nama = extract_nama_wajib_pajak(
                file_bytes,
                [pages[0]],   # halaman awal STP
                images,
                headers,
                debug=debug
            )

            if not nama or not nama.strip():
                nama = FALLBACK_NAME

        else:
            # ✅ SKPKB / SKPN / SKPLB: cache lintas lampiran
            if cached_nama_wp is None:
                nama = extract_nama_wajib_pajak(
                    file_bytes,
                    [pages[0]],
                    images,
                    headers,
                    debug=debug
                )

                if nama and nama.strip():
                    cached_nama_wp = nama.strip()
                else:
                    cached_nama_wp = FALLBACK_NAME

            nama = cached_nama_wp

        # =====================================
        # SANITIZE NAMA (AMAN)
        # =====================================
        safe_nama = re.sub(r'[\\/:"*?<>|]+', '', nama)
        safe_nama = re.sub(r'\s+', '_', safe_nama).upper()

        nomor_for_name = nomor if nomor else FALLBACK_NAME
        final_name = f"{nomor_for_name}_{safe_nama}.pdf"

        results.append((final_name, out_bytes))

    return results

             
# ---------------------------------------------------------------------------------
# ---------------- Streamlit UI (header + sidebar + tabs) -------------------------
# =========================
# LOGIN GATE (WAJIB)
# =========================
if not st.session_state.logged_in:
    login_page()
    st.stop()

st.set_page_config(page_title="SIPANDA-OCR — Sistem Pemisahan dan Penamaan Dokumen berbasis OCR", layout="wide")
st.markdown("""<style>
section[data-testid="stSidebar"] {
    background-color: #002b5c !important;
}
</style>""", unsafe_allow_html=True)

# Custom sidebar styling
st.markdown("""
    <style>
        /* ===== LOGOUT BUTTON ===== */ 
        /* Sidebar background */
        section[data-testid="stSidebar"] {
            background-color: #002b5c !important;
            padding-top: 20px;
        }

        /* Remove sidebar border */
        section[data-testid="stSidebar"] > div {
            border-right: none !important;
        }

        /* Text color inside sidebar */
        section[data-testid="stSidebar"] * {
            color: white !important;
        }

        /* Debug checkbox styling */
        .stCheckbox label {
            color: white !important;
        }

        /* Divider line color */
        hr {
            border-top: 1px solid #ffffff55 !important;
        }
            
        /* ===== SharePoint Rectangle Box ===== */
        section[data-testid="stSidebar"] .sidebar-box {
            border: 1.5px solid #FFD200;
            border-radius: 8px;
            padding: 12px 14px;
            margin-top: 18px;
            background-color: rgba(255, 255, 255, 0.15);
        }

        /* Sidebar box title */
        section[data-testid="stSidebar"] .sidebar-box h4 {
            margin: 0 0 6px 0;
            font-size: 1rem;
            font-weight: 600;
        }
        /* Sidebar box text */
        section[data-testid="stSidebar"] .sidebar-box p {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.95;
            
        /* Hide real Streamlit logout button */
        button[kind="secondary"][aria-label="Logout"] {
            opacity: 0;
            height: 0;
            padding: 0;
            margin: 0;
}
}    
    </style>
""", unsafe_allow_html=True)

# Inject theme CSS (header, tabs, buttons)
# ---------- UI: Header (no logo) + Sidebar logo only + Tab hover yellow ----------
# Minimal header (no logo in main page) — clean hero bar
import streamlit as st

import streamlit as st

COLOR_BLUE = "#002b5c"
COLOR_YELLOW = "#FFD200"
TAB_WIDTH = 140

st.markdown(f"""
<style>

.hero {{
  background: linear-gradient(90deg, {COLOR_BLUE} 0%, {COLOR_BLUE} 100%);
  padding: 18px 22px;
  border-radius: 10px;
  color: #ffffff;
  display:flex;
  align-items:center;
  justify-content:space-between;
  box-shadow: 0 8px 24px rgba(2,12,34,0.06);
  margin-bottom: 18px;
}}
.hero h1 {{
  margin:0;
  font-size:25px;
  font-weight:700;
}}
.hero p {{
  margin:0;
  color:#d7e7ff;
  font-size:15px;
}}

.hero-divider {{
    width: 1000px;
    height: 4px;
    background-color: #FFD200;
    margin-top: 12px;
    margin-bottom: 6px;
    border-radius: 2px;
}}

[data-testid="stSidebar"] .sidebar-logo {{
  display:flex;
  align-items:center;
  justify-content:center;
  padding:12px 4px;
  margin-bottom:6px;
}}
[data-testid="stSidebar"] .sidebar-logo img {{
  width:160px;
  border-radius:10px;
  background:#fff;
  padding:6px;
  box-shadow: 0 6px 18px rgba(2,12,34,0.06);
}}

/* TABLIST */
.stTabs [role="tablist"] {{
  display: flex !important;
  gap: 0 !important;
  overflow-x: auto;
  padding: 0 4px;
}}

.stTabs [role="tablist"] button {{
  width: {TAB_WIDTH}px !important;
  min-width: {TAB_WIDTH}px !important;
  max-width: {TAB_WIDTH}px !important;

  border-radius: 10px 10px 0 0 !important;
  margin: 0 8px !important;
  padding: 10px 6px !important;
  font-weight:700 !important;
  border: none !important;
  background: rgba(0,0,0,0.03) !important;
  transition: background-color 120ms ease, color 120ms ease, transform 120ms ease;
  color: #222 !important;
  font-size:20px !important;
  display:inline-flex !important;
  align-items:center !important;
  justify-content:center !important;
  box-sizing: border-box !important;
}}

.stTabs [role="tablist"] button:hover {{
  background: {COLOR_BLUE} !important;
  color: {COLOR_YELLOW} !important;
  transform: translateY(-2px);
}}

.stTabs [role="tablist"] button[aria-selected="true"] {{
  background: {COLOR_YELLOW} !important;
  color: {COLOR_BLUE} !important;
  box-shadow: 0 8px 20px rgba(0,0,0,0.08) !important;
  position: relative;
}}

.stTabs [role="tablist"] button[aria-selected="true"]::after {{
  content: "";
  position: absolute;
  left: 10px;
  right: 10px;
  bottom: -3px;
  height: 3px;
  background: #cc0000;
  border-radius: 2px;
}}

.stTabs [role="tablist"] button span,
.stTabs [role="tablist"] button p {{
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin: 0;
  padding: 0 6px;
}}

/* Paksa teks tab menjadi bold & ukuran lebih besar */
.stTabs [role="tablist"] button p,
.stTabs [role="tablist"] button span,
.stTabs [role="tablist"] button {{
    font-weight: 800 !important;   /* Super bold */
    font-size: 16px !important;    /* Perbesar ukuran */
    color: inherit !important;     /* Ikuti warna aktif/non-aktif */
}}

.stButton>button {{
  background: {COLOR_BLUE} !important;
  color: white !important;
  border-radius:8px !important;
  padding:8px 12px !important;
  font-weight:600;
}}
.stDownloadButton>button {{
  background: {COLOR_YELLOW} !important;
  color: {COLOR_BLUE} !important;
  border-radius:8px !important;
  padding:8px 12px !important;
  font-weight:700;
}}

.panel {{
  background: #ffffff;
  padding: 16px;
  border-radius: 10px;
  box-shadow: 0 6px 18px rgba(2,12,34,0.04);
}}

</style>
""", unsafe_allow_html=True)

# Hero header (no logo here)
st.markdown(f"""
<div class="hero">
  <div>
    <h1>SIPANDA-OCR — Sistem Pemisahan dan Penamaan Dokumen berbasis OCR</h1>
    <p>Pemisahan dan penamaan dokumen otomatis berbasis OCR untuk STP, SKPKB, SKPLB, dan SKPN</p>
    <div style="opacity:0.9; font-size:13px; color:#e6f0ff">Dibuat untuk KPP Pondok Aren © 2025</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Sidebar: only DJP logo + debug + tips
# local file path
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>", unsafe_allow_html=True)
    LOGO_PATH = "img/logoponaren_white.png"

    try:
        st.image(LOGO_PATH)
    except Exception as e:
        st.error(f"Gagal load logo: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# Sidebar login
    st.success(f"👤 Login sebagai: {st.session_state.username}")

# ==========================================================
    st.markdown("---")
    st.checkbox("Debug mode (show OCR snippets)", value=False, key="debug_sidebar")
    st.markdown("**Tips**: Aktifkan debug saat testing untuk melihat OCR kandidat dan crop preview.")
    st.markdown("""
    <div class="sidebar-box">
        <h4>☁️ SharePoint</h4>
        <a class="sidebar-link" href="https://kemenkeu.sharepoint.com/:u:/s/Pelita4532/IQC_ijJCdb1bS41b8e0VQEUqAT5nNB7lH8jrDgF6Z4SBe_8?e=m3JW5o" target="_blank">
            📤 Upload hasil ke SharePoint
        </a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")


# logo copyright
    st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://github.com/ranydwi">ranydwi</a></h6>',
            unsafe_allow_html=True,
        )        


# Read debug flag from sidebar
debug = st.session_state.get("debug_sidebar", False)

# Tabs (Overview + types)

tabs = st.tabs(["Overview","STP","SKPKB","SKPLB","SKPN"])

# Overview tab
with tabs[0]:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Dashboard Overview")
    st.markdown("""
    **SIPANDA-OCR** adalah dashboard automasi dokumen perpajakan yang dirancang untuk menyederhanakan proses pengolahan arsip secara cepat, akurat, dan terstandar. 
    Sistem ini mengintegrasikan pemisahan halaman otomatis, ekstraksi informasi berbasis OCR, dan 
    penamaan dokumen yang konsisten untuk berbagai jenis surat perpajakan, termasuk STP, SKPKB, SKPLB, dan SKPN. 
    Dengan pendekatan yang terpusat dan cerdas, CORE-DOC mendukung percepatan digitalisasi arsip 
    serta meningkatkan efisiensi dan kualitas tata kelola dokumen di lingkungan KPP.
    """)
    st.markdown("----")
    st.subheader("Alur Kerja (Workflow)")
    st.image("img\Flow CORE-DOC.png")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("----")
    st.subheader("Panduan Sistem Arsip Digital Perpajakan")
    st.markdown("Gunakan panduan ini sebagai acuan dalam mengelola arsip dokumen perpajakan secara digital, mulai dari unggah, pemrosesan, hingga penyimpanan akhir.")
    st.image("assets/1.png", width = 250)
    with open("assets\Panduan Penggunaan Sistem Arsip Digital Pondok Aren.pdf", "rb") as f:
        st.download_button(
            label="📘 Download Panduan Lengkap (PDF)",
            data=f,
            file_name="Panduan Penggunaan Sistem Arsip Digital Pondok Aren.pdf",
            mime="application/pdf"
        )


# STP tab
with tabs[1]:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.header("Mode: Surat Tagihan Pajak (STP)")
    st.markdown("""
    <div style="border-left:5px solid #FFD200; padding:12px; background-color:#fff8e5">
    <b>📄 Spesifikasi Otomatisasi:</b>
    <ul>
    <li>2 halaman per STP</li>
    <li>Halaman billing diikutsertakan (jika ada)</li>
    <li>Pemisahan otomatis berdasarkan struktur dokumen</li>
    </ul>

    <b>⚠️ Pastikan:</b>
    <ul>
    <li>Urutan halaman benar</li>
    <li>Tidak digabung dengan jenis surat lain</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""#### Upload PDF(s) untuk STP""")
    uploaded = st.file_uploader(
    "Upload",
    type="pdf",
    accept_multiple_files=True,
    key="stp_up",
    label_visibility="collapsed"
)

    if uploaded:
        all_out = []
        prog = st.progress(0)
        for i,f in enumerate(uploaded):
            b = f.read()
            try:
                parts = process_for_type(b, f.name, "STP", debug=debug)
                all_out.extend(parts)
                st.success(f"{len(parts)} parts created from {f.name}")
            except Exception as e:
                st.error(f"Error processing {f.name}: {e}")
            prog.progress(int((i+1)/len(uploaded)*100))
        if all_out:
            st.subheader("Generated files:")
            for name,_ in all_out:
                st.write("-", name)
            buf = BytesIO()
            with zipfile.ZipFile(buf,"w") as z:
                for name,bts in all_out:
                    z.writestr(name,bts)
            buf.seek(0)
            st.download_button("Download ZIP (STP)", buf, file_name="stp_results.zip")
    st.markdown("</div>", unsafe_allow_html=True)

# SKPKB tab
with tabs[2]:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.header("Mode: Surat Ketetapan Pajak Kurang Bayar (SKPKB)")
    st.markdown("""
    <div style="border-left:5px solid #FFD200; padding:12px; background-color:#fff8e5">
    <b>📄 Spesifikasi Otomatisasi:</b>
    <ul>
    <li>2 halaman untuk SKPKB non-PPN</li>
    <li>4 halaman khusus untuk SKPKB PPN</li>
    <li>Pemisahan dokumen dilakukan otomatis sesuai struktur SKPKB</li>
    </ul>

    <b>⚠️ Pastikan:</b>
    <ul>
    <li>Jenis pajak (PPN / non-PPN) sudah sesuai</li>
    <li>Tidak digabung dengan jenis surat lain (STP, SKPN, SKPLB)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""#### Upload PDF(s) untuk SKPKB""")
    uploaded = st.file_uploader(
    "Upload",
    type="pdf",
    accept_multiple_files=True,
    key="skpkb_up",
    label_visibility="collapsed"
)
    if uploaded:
        all_out=[]
        prog=st.progress(0)
        for i,f in enumerate(uploaded):
            b=f.read()
            try:
                parts = process_for_type(b, f.name, "SKPKB", debug=debug)
                all_out.extend(parts)
                st.success(f"{len(parts)} parts created from {f.name}")
            except Exception as e:
                st.error(f"Error processing {f.name}: {e}")
            prog.progress(int((i+1)/len(uploaded)*100))
        if all_out:
            st.subheader("Generated files:")
            for name,_ in all_out:
                st.write("-", name)
            buf = BytesIO()
            with zipfile.ZipFile(buf,"w") as z:
                for name,bts in all_out:
                    z.writestr(name,bts)
            buf.seek(0)
            st.download_button("Download ZIP (SKPKB)", buf, file_name="skpkb_results.zip")
    st.markdown("</div>", unsafe_allow_html=True)

# SKPLB tab
with tabs[3]:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.header("Mode: Surat Ketetapan Pajak Lebih Bayar (SKPLB)")
    st.markdown("""
    <div style="border-left:5px solid #FFD200; padding:12px; background-color:#fff8e5">
    <b>📄 Spesifikasi Otomatisasi:</b>
    <ul>
    <li>Seluruh halaman SKPLB wajib diikutsertakan</li>
    <li>Lampiran-lampiran (apabila ada) wajib ikut discan</li>
    <li>Pemisahan dokumen dilakukan berdasarkan kop surat dan nomor surat</li>
    </ul>

    <b>⚠️ Perhatian Khusus:</b>
    <ul>
    <li>Nomor surat harus terlihat jelas untuk mendukung OCR</li>
    <li>Tidak ada halaman atau lampiran yang terlewat</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""#### Upload PDF(s) untuk SKPLB""")
    uploaded = st.file_uploader(
    "Upload",
    type="pdf",
    accept_multiple_files=True,
    key="skplb_up",
    label_visibility="collapsed"
)
    if uploaded:
        all_out=[]
        prog=st.progress(0)
        for i,f in enumerate(uploaded):
            b=f.read()
            try:
                parts = process_for_type(b, f.name, "SKPLB", debug=debug)
                all_out.extend(parts)
                st.success(f"{len(parts)} parts created from {f.name}")
            except Exception as e:
                st.error(f"Error processing {f.name}: {e}")
            prog.progress(int((i+1)/len(uploaded)*100))
        if all_out:
            st.subheader("Generated files:")
            for name,_ in all_out:
                st.write("-", name)
            buf = BytesIO()
            with zipfile.ZipFile(buf,"w") as z:
                for name,bts in all_out:
                    z.writestr(name,bts)
            buf.seek(0)
            st.download_button("Download ZIP (SKPLB)", buf, file_name="skplb_results.zip")
    st.markdown("</div>", unsafe_allow_html=True)

# SKPN tab
with tabs[4]:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.header("Mode: Surat Ketetapan Pajak Nihil (SKPN)")
    st.markdown("""
    <div style="border-left:5px solid #FFD200; padding:12px; background-color:#fff8e5">
    <b>📄 Spesifikasi Otomatisasi:</b>
    <ul>
    <li>2 halaman untuk SKPN non-PPN</li>
    <li>4 halaman khusus untuk SKPN PPN</li>
    <li>Pemisahan dilakukan otomatis berdasarkan struktur SKPN</li>
    </ul>

    <b>⚠️ Pastikan:</b>
    <ul>
    <li>Seluruh halaman utama tercantum lengkap</li>
    <li>Tidak terdapat halaman dari jenis surat lain</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""#### Upload PDF(s) untuk SKPN""")
    uploaded = st.file_uploader(
    "Upload",
    type="pdf",
    accept_multiple_files=True,
    key="skpn_up",
    label_visibility="collapsed"
)
    if uploaded:
        all_out=[]
        prog=st.progress(0)
        for i,f in enumerate(uploaded):
            b=f.read()
            try:
                parts = process_for_type(b, f.name, "SKPN", debug=debug)
                all_out.extend(parts)
                st.success(f"{len(parts)} parts created from {f.name}")
            except Exception as e:
                st.error(f"Error processing {f.name}: {e}")
            prog.progress(int((i+1)/len(uploaded)*100))
        if all_out:
            st.subheader("Generated files:")
            for name,_ in all_out:
                st.write("-", name)
            buf = BytesIO()
            with zipfile.ZipFile(buf,"w") as z:
                for name,bts in all_out:
                    z.writestr(name,bts)
            buf.seek(0)
            st.download_button("Download ZIP (SKPN)", buf, file_name="skpn_results.zip")
    st.markdown("</div>", unsafe_allow_html=True)
