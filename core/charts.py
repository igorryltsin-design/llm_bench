"""Графики в оформлении для статьи: Times New Roman 12 пт, точная ширина в см, подпись «Рис. N. …»,
десятичная запятая, цветной или чёрно-белый вариант. Сохранение: PNG 300 dpi (в Word вставляется
в натуральную величину, шрифт остаётся 12 пт), SVG (текст — текстом), PDF, DOCX (рисунок и подпись
отдельным абзацем)."""
import io
import logging
import math
from dataclasses import dataclass, replace

import matplotlib

matplotlib.use('Agg')
import numpy as np                                     # noqa: E402
from matplotlib import font_manager                     # noqa: E402
from matplotlib.figure import Figure                    # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

CM = 1 / 2.54
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948',
          '#6b7280', '#0e7490']
GRAYS = ['#303030', '#8c8c8c', '#d0d0d0', '#5e5e5e', '#b0b0b0', '#ffffff', '#1a1a1a', '#a0a0a0']
HATCHES = ['', '///', '\\\\\\', 'xxx', '...', '---', '+++', 'ooo']
LINESTYLES = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
_AVAILABLE = None


def font_names():
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE = sorted({f.name for f in font_manager.fontManager.ttflist})
    return _AVAILABLE


@dataclass(frozen=True)
class Style:
    font: str = 'Times New Roman'
    size: float = 12.0
    width_cm: float = 16.0
    bw: bool = False
    caption: str = ''
    number: str = ''
    with_caption: bool = True
    decimal_comma: bool = True

    def family(self):
        have = set(font_names())
        fam = [f for f in dict.fromkeys([self.font, 'Times New Roman', 'Times', 'Liberation Serif']) if f in have]
        return fam + ['DejaVu Serif']

    def caption_text(self):
        if not self.caption:
            return ''
        return f'Рис. {self.number}. {self.caption}' if str(self.number).strip() else self.caption

    def color(self, i):
        return GRAYS[i % len(GRAYS)] if self.bw else COLORS[i % len(COLORS)]

    def hatch(self, i):
        return HATCHES[i % len(HATCHES)] if self.bw else ''

    def ls(self, i):
        return LINESTYLES[i % len(LINESTYLES)] if self.bw else '-'

    def marker(self, i):
        return MARKERS[i % len(MARKERS)]

    def cmap(self):
        return 'Greys' if self.bw else 'Blues'


def rc(style: Style):
    return {
        'font.family': 'serif', 'font.serif': style.family(), 'font.size': style.size,
        'axes.titlesize': style.size, 'axes.labelsize': style.size, 'xtick.labelsize': style.size,
        'ytick.labelsize': style.size, 'legend.fontsize': style.size, 'figure.titlesize': style.size,
        'mathtext.fontset': 'custom', 'mathtext.rm': style.font, 'mathtext.it': f'{style.font}:italic',
        'mathtext.bf': f'{style.font}:bold', 'axes.unicode_minus': True, 'svg.fonttype': 'none',
        'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': '#d9d9d9', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'legend.frameon': False, 'lines.linewidth': 1.6, 'axes.linewidth': 0.8,
        'hatch.linewidth': 0.6, 'patch.edgecolor': '#303030' if style.bw else 'none',
    }


def num(x, digits=2, style: Style = None):
    """Число с десятичной запятой (по умолчанию)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '—'
    s = f'{x:.{digits}f}'.replace('-', '−')
    return s.replace('.', ',') if (style is None or style.decimal_comma) else s


def comma_axes(ax, style, which='both'):
    if not style.decimal_comma:
        return
    fmt = FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ',').replace('-', '−'))
    if which in ('x', 'both'):
        ax.xaxis.set_major_formatter(fmt)
    if which in ('y', 'both'):
        ax.yaxis.set_major_formatter(fmt)


def wrap(text, width_pt, style: Style):
    """Перенос подписи по фактической ширине строки в пунктах."""
    from matplotlib.textpath import TextToPath
    ttp = TextToPath()
    prop = font_manager.FontProperties(family=style.family(), size=style.size)

    def w(s):
        return ttp.get_text_width_height_descent(s, prop, ismath=False)[0]

    lines, cur = [], ''
    for word in text.split():
        cand = f'{cur} {word}'.strip()
        if cur and w(cand) > width_pt:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


class Canvas:
    """Фигура с зарезервированным местом под подпись."""

    def __init__(self, style: Style, height_cm: float, nrows=1, ncols=1, polar=False, **sub_kw):
        self.style = style
        matplotlib.rcParams.update(rc(style))
        self.cap_lines = []
        cap = style.caption_text() if style.with_caption else ''
        w_in = style.width_cm * CM
        cap_h = 0.0
        if cap:
            self.cap_lines = wrap(cap, style.width_cm * CM * 72 - 6, style)
            cap_h = (len(self.cap_lines) * style.size * 1.25 + style.size * 0.9) / 72
        h_in = height_cm * CM + cap_h
        self.fig = Figure(figsize=(w_in, h_in), dpi=100)
        self.fig.set_layout_engine('constrained', rect=(0, cap_h / h_in, 1, 1 - cap_h / h_in), h_pad=0.04, w_pad=0.04)
        kw = dict(sub_kw)
        if polar:
            kw['subplot_kw'] = {'projection': 'polar'}
        self.axes = self.fig.subplots(nrows, ncols, squeeze=False, **kw)
        self.cap_h, self.h_in = cap_h, h_in

    @property
    def ax(self):
        return self.axes[0][0]

    def finish(self):
        if self.cap_lines:
            y = (self.style.size * 0.3 / 72) / self.h_in
            self.fig.text(0.5, y, '\n'.join(self.cap_lines), ha='center', va='bottom', fontsize=self.style.size,
                          family=self.style.family(), linespacing=1.25)
        return self.fig


# ---------------- экспорт ----------------
def to_bytes(fig, fmt='png', dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, metadata=None if fmt != 'png' else {'Software': 'LLM Bench'})
    return buf.getvalue()


def _docx_base():
    from docx import Document
    from docx.enum.style import WD_STYLE_TYPE  # noqa: F401
    from docx.oxml.ns import qn
    from docx.shared import Pt
    doc = Document()
    st = doc.styles['Normal']
    st.font.name = 'Times New Roman'
    st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    st.element.rPr.rFonts.set(qn('w:cs'), 'Times New Roman')
    pf = st.paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(0), 1.0
    sec = doc.sections[0]
    from docx.shared import Cm
    sec.left_margin, sec.right_margin = Cm(2.0), Cm(1.5)
    return doc


def add_figure_docx(doc, png: bytes, width_cm: float, caption: str):
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(io.BytesIO(png), width=Cm(width_cm))
    if caption:
        c = doc.add_paragraph(caption)
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.space_after = Pt(12)


def add_table_docx(doc, df, caption: str, decimal_comma=True, digits=2):
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.shared import Pt
    if caption:
        c = doc.add_paragraph(caption)
        c.paragraph_format.space_before = Pt(6)
        c.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, col in enumerate(df.columns):
        t.rows[0].cells[i].text = str(col)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            col = df.columns[i]
            dg = digits.get(col, digits.get('*', 2)) if isinstance(digits, dict) else digits
            if isinstance(v, float):
                s = num(v, dg) if decimal_comma else f'{v:.{dg}f}'
            else:
                s = '—' if v is None else str(v)
            cells[i].text = s
    for row in t.rows:
        for cell in row.cells:
            for par in cell.paragraphs:
                for run in par.runs:
                    run.font.size = Pt(12)
                    run.font.name = 'Times New Roman'
    doc.add_paragraph()


def figure_docx(png: bytes, width_cm: float, caption: str) -> bytes:
    doc = _docx_base()
    add_figure_docx(doc, png, width_cm, caption)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def table_docx(df, caption: str, digits=2, decimal_comma=True) -> bytes:
    doc = _docx_base()
    add_table_docx(doc, df, caption, decimal_comma, digits)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def report_docx(parts) -> bytes:
    """parts: [('h', текст) | ('p', текст) | ('fig', png, width_cm, подпись) | ('table', df, подпись, digits)]."""
    from docx.shared import Pt
    doc = _docx_base()
    for part in parts:
        if part[0] == 'h':
            p = doc.add_paragraph()
            r = p.add_run(part[1])
            r.bold = True
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(6)
        elif part[0] == 'p':
            doc.add_paragraph(part[1])
        elif part[0] == 'fig':
            add_figure_docx(doc, part[1], part[2], part[3])
        elif part[0] == 'table':
            add_table_docx(doc, part[1], part[2], True, part[3] if len(part) > 3 else 2)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---------------- типовые графики ----------------
def wrap_label(s, width=26):
    import textwrap
    s = str(s)
    return '\n'.join(textwrap.wrap(s, width, break_long_words=False)) if len(s) > width else s


def _legend(ax, style, n, loc='best', ncol=None, outside=True):
    if n <= 1:
        return
    if outside:
        labels = ax.get_legend_handles_labels()[1]
        longest = max((len(x) for x in labels), default=0)
        width_chars = style.width_cm * 4.2                       # ≈ символов 12 пт в строке заданной ширины
        auto = max(1, min(n, int(width_chars // (longest + 6))))
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=ncol or auto, handlelength=1.8,
                  columnspacing=1.2, borderaxespad=0.3)
    else:
        ax.legend(loc=loc, ncol=ncol or 1)


def bars(style: Style, categories, series: dict, errors: dict = None, ylabel='', height_cm=8.0, ylim=None,
         horizontal=False, value_labels=False, digits=1):
    """Сгруппированные столбцы. series: {серия: [значения по категориям]}; errors: {серия: [(lo, hi)]}."""
    names = list(series)
    categories = [wrap_label(c, 30 if horizontal else 16) for c in categories]
    n_cat, n_ser = len(categories), max(1, len(names))
    auto_h = max(height_cm, 1.2 + 0.55 * n_cat * max(1, n_ser * 0.6)) if horizontal else height_cm
    cv = Canvas(style, auto_h)
    ax = cv.ax
    width = 0.8 / n_ser
    x = np.arange(n_cat)
    for i, s in enumerate(names):
        vals = np.array([np.nan if v is None else v for v in series[s]], float)
        pos = x - 0.4 + width * (i + 0.5)
        err = None
        if errors and errors.get(s):
            lo = np.array([np.nan if e is None or e[0] is None else e[0] for e in errors[s]], float)
            hi = np.array([np.nan if e is None or e[1] is None else e[1] for e in errors[s]], float)
            err = np.vstack([np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)])
            err = np.nan_to_num(err)
        kw = dict(color=style.color(i), hatch=style.hatch(i), label=s,
                  edgecolor='#303030' if style.bw else 'none', linewidth=0.6 if style.bw else 0)
        if horizontal:
            bc = ax.barh(pos, np.nan_to_num(vals), height=width, xerr=err, capsize=2.5,
                         error_kw=dict(elinewidth=0.8, ecolor='#404040'), **kw)
        else:
            bc = ax.bar(pos, np.nan_to_num(vals), width=width, yerr=err, capsize=2.5,
                        error_kw=dict(elinewidth=0.8, ecolor='#404040'), **kw)
        if value_labels:
            ax.bar_label(bc, labels=[num(v, digits, style) if not np.isnan(v) else '' for v in vals],
                         padding=2, fontsize=style.size)
    if horizontal:
        ax.set_yticks(x, categories)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.grid(axis='y', visible=False)
        if ylim:
            ax.set_xlim(*ylim)
        comma_axes(ax, style, 'x')
    else:
        ax.set_xticks(x, categories)
        ax.set_ylabel(ylabel)
        ax.grid(axis='x', visible=False)
        if ylim:
            ax.set_ylim(*ylim)
        comma_axes(ax, style, 'y')
        if n_cat > 4 or max((len(str(c)) for c in categories), default=0) > 14:
            for lab in ax.get_xticklabels():
                lab.set_rotation(30)
                lab.set_ha('right')
    _legend(ax, style, n_ser)
    return cv.finish()


def heatmap(style: Style, matrix, row_labels, col_labels, cbar_label='', height_cm=None, vmin=None, vmax=None,
            digits=0, annotate=True, xlabel='', ylabel=''):
    M_ = np.array(matrix, float)
    row_labels = [wrap_label(r, 28) for r in row_labels]
    col_labels = [wrap_label(c, 16) for c in col_labels]
    rotate = len(col_labels) > 5 or max((len(str(c)) for c in col_labels), default=0) > 10
    longest_col = max((len(x) for c in col_labels for x in str(c).split('\n')), default=0)
    h = height_cm or max(6.0, 2.0 + 0.95 * len(row_labels) + (0.2 * longest_col if rotate else 0))
    cv = Canvas(style, h)
    ax = cv.ax
    ax.grid(False)
    diverging = vmin is not None and vmin < 0 < (vmax if vmax is not None else 1)
    cmap = ('Greys' if style.bw else 'RdBu') if diverging else style.cmap()
    im = ax.imshow(M_, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for lab in ax.get_xticklabels():
        if rotate:
            lab.set_rotation(40)
            lab.set_ha('right')
    for s in ax.spines.values():
        s.set_visible(False)
    if annotate:
        lo, hi = np.nanmin(M_) if vmin is None else vmin, np.nanmax(M_) if vmax is None else vmax
        for i in range(M_.shape[0]):
            for j in range(M_.shape[1]):
                v = M_[i, j]
                if np.isnan(v):
                    continue
                rel = (v - lo) / ((hi - lo) or 1)
                dark = (rel < 0.2 or rel > 0.8) if diverging and not style.bw else rel > 0.55
                ax.text(j, i, num(v, digits, style), ha='center', va='center', fontsize=style.size,
                        color='white' if dark else 'black')
    cb = cv.fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label(cbar_label)
    cb.outline.set_visible(False)
    if style.decimal_comma:
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ',')))
    return cv.finish()


def lines(style: Style, x, series: dict, dashed: dict = None, xlabel='', ylabel='', height_cm=8.0, ylim=None,
          xticks=None, legend_note=''):
    """Линии с маркерами; dashed — штриховые линии без маркеров (например, центроид)."""
    cv = Canvas(style, height_cm)
    ax = cv.ax
    for i, (name, ys) in enumerate(series.items()):
        c = style.color(i)
        ax.plot(x, ys, marker=style.marker(i), linestyle=style.ls(i), color=c, label=name, markersize=5,
                markerfacecolor='white' if style.bw and i % 2 else c)
        if dashed and name in dashed:
            ax.axhline(dashed[name], color=c, linestyle=(0, (4, 3)), linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if ylim:
        ax.set_ylim(*ylim)
    comma_axes(ax, style, 'y')
    _legend(ax, style, len(series))
    return cv.finish()


def scatter(style: Style, points, xlabel='', ylabel='', height_cm=9.0, highlight=None, xlim=None, ylim=None,
            size_label=''):
    """points: [(x, y, подпись, размер или None)]; highlight — индексы точек парето-фронта."""
    cv = Canvas(style, height_cm)
    ax = cv.ax
    sizes = [p[3] for p in points if p[3]]
    smax = max(sizes) if sizes else 1
    for i, (x, y, lab, s) in enumerate(points):
        ms = 40 + 260 * (s / smax) if s else 60
        hl = highlight is not None and i in highlight
        ax.scatter([x], [y], s=ms, color=style.color(i), marker=style.marker(i) if style.bw else 'o',
                   edgecolor='black' if hl else ('#303030' if style.bw else 'white'), linewidth=1.6 if hl else 0.6,
                   zorder=3, alpha=0.9)
        ax.annotate(lab, (x, y), xytext=(6, 5), textcoords='offset points', fontsize=style.size)
    if highlight and len(highlight) > 1:
        pts = sorted([points[i] for i in highlight], key=lambda p: p[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], linestyle=(0, (4, 3)), color='#606060', linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    comma_axes(ax, style)
    return cv.finish()


def boxes(style: Style, data: dict, ylabel='', height_cm=8.0):
    cv = Canvas(style, height_cm)
    ax = cv.ax
    names = list(data)
    bp = ax.boxplot([data[n] for n in names], tick_labels=names, patch_artist=True, widths=0.55,
                    medianprops=dict(color='black', linewidth=1.4), flierprops=dict(marker='o', markersize=3))
    for i, b in enumerate(bp['boxes']):
        b.set_facecolor(style.color(i) if not style.bw else GRAYS[(i + 2) % len(GRAYS)])
        b.set_hatch(style.hatch(i))
        b.set_alpha(0.85)
    ax.set_ylabel(ylabel)
    ax.grid(axis='x', visible=False)
    comma_axes(ax, style, 'y')
    if len(names) > 3:
        for lab in ax.get_xticklabels():
            lab.set_rotation(25)
            lab.set_ha('right')
    return cv.finish()


def radar(style: Style, labels, series: dict, height_cm=10.0):
    """Лепестковая диаграмма нормированных показателей g ∈ [0, 1]."""
    cv = Canvas(style, height_cm, polar=True)
    ax = cv.ax
    n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    ang += ang[:1]
    for i, (name, vals) in enumerate(series.items()):
        v = list(vals) + [vals[0]]
        ax.plot(ang, v, color=style.color(i), linestyle=style.ls(i), marker=style.marker(i), markersize=4, label=name)
        if not style.bw:
            ax.fill(ang, v, color=style.color(i), alpha=0.08)
    ax.set_xticks(ang[:-1], labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.5, 0.8], [num(0.2, 1, style), num(0.5, 1, style), num(0.8, 1, style)])
    ax.spines['polar'].set_visible(False)
    _legend(ax, style, len(series), outside=False, loc='upper right')
    if len(series) > 1:
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05))
    return cv.finish()


def fuzzy_terms(style: Style, in_terms: dict, out_terms: dict, height_cm=6.0):
    """Рис. 3 статьи: а — входные термы на нормированной шкале; б — выходные термы «Качество»."""
    cv = Canvas(style, height_cm, 1, 2)
    a, b = cv.axes[0]
    g = np.linspace(0, 1, 501)
    from core.fuzzy import trap
    for i, (t, p) in enumerate(in_terms.items()):
        a.plot(g, trap(g, *p), color=style.color(i) if not style.bw else 'black', linestyle=style.ls(i) if style.bw else '-')
        peak = g[np.argmax(trap(g, *p))] if t != 'С' else p[1]
        a.text({'Н': 0.1, 'С': 0.5, 'В': 0.9}.get(t, peak), 1.04, t, ha='center', va='bottom')
    a.set_xlabel('g')
    a.set_ylabel('μ', rotation=0, labelpad=10)
    a.set_ylim(0, 1.15)
    a.set_title('а', loc='center', y=-0.42)
    q = np.linspace(0, 10, 1001)
    for i, (t, p) in enumerate(out_terms.items()):
        b.plot(q, trap(q, *p), color=style.color(i) if not style.bw else 'black',
               linestyle=style.ls(i) if style.bw else '-')
        b.text(p[1] if p[1] >= 0 else 0, 1.04, t, ha='center', va='bottom')
    b.set_xlabel('Качество')
    b.set_ylim(0, 1.15)
    b.set_title('б', loc='center', y=-0.42)
    for ax in (a, b):
        comma_axes(ax, style)
        ax.grid(alpha=0.5)
    return cv.finish()


def rule_clipped(style: Style, model, raw: dict, k=2, height_cm=7.0):
    """Рис. 4 статьи: усечённые заключения сработавших правил, центроид и ВСММ."""
    from core import fuzzy as F
    g, mu, res = model.infer(raw)
    act = [r for r in res if r['w'] > 0]
    qc, agg = F.q_centroid(res)
    qw = F.q_wmsm(res, k)
    cv = Canvas(style, height_cm)
    ax = cv.ax
    for t, p in model.out_terms.items():
        ax.plot(F.QGRID, F.trap(F.QGRID, *p), color='#b0b0b0', linewidth=0.8)
        ax.text(max(p[1], 0), 1.03, t, ha='center', va='bottom', color='#606060', zorder=5,
                bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
    for i, r in enumerate(act):
        lab = f"{r['rule']} (ω = {num(r['w'], 2, style)}, {r['term']})"
        if style.bw:
            ax.fill_between(F.QGRID, r['clipped'], facecolor='none', hatch=HATCHES[(i + 1) % len(HATCHES)],
                            edgecolor='#505050', linewidth=0, label=lab)
        else:
            ax.fill_between(F.QGRID, r['clipped'], color=style.color(i), alpha=0.35, linewidth=0, label=lab)
    ax.plot(F.QGRID, agg, color='black', linewidth=1.0)
    ax.axvline(qc, color='black', linestyle=(0, (4, 3)), linewidth=1.2, label=f'центроид: {num(qc, 2, style)}')
    ax.axvline(qw, color='black', linestyle='-', linewidth=2.0, label=f'ВСММ (k = {k}): {num(qw, 2, style)}')
    ymax = 1.12
    ax.set_xlim(0, 10)
    ax.set_ylim(0, ymax)
    ax.set_xlabel('Качество')
    ax.set_ylabel('μ', rotation=0, labelpad=10)
    comma_axes(ax, style)
    _legend(ax, style, len(act) + 2, ncol=2)
    return cv.finish()


def training_curve(style: Style, train: list, val: list, height_cm=7.0):
    """train/val: [(итерация, потери)]."""
    cv = Canvas(style, height_cm)
    ax = cv.ax
    if train:
        ax.plot(*zip(*train), color=style.color(0), linestyle='-', label='обучающая выборка')
    if val:
        ax.plot(*zip(*val), color=style.color(1), linestyle='--', marker='o', markersize=4,
                label='проверочная выборка')
    ax.set_xlabel('Итерация')
    ax.set_ylabel('Функция потерь')
    comma_axes(ax, style, 'y')
    _legend(ax, style, int(bool(train)) + int(bool(val)))
    return cv.finish()


def with_style(style: Style, **kw):
    return replace(style, **kw)
