from __future__ import annotations

import base64
import html
import io
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional

import emoji
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

try:
    import ftfy
except Exception:  # pragma: no cover
    ftfy = None


SHOW_BASIC_STATS = True
SHOW_REPLY_TIMES = True
SHOW_LEXICAL = True
SHOW_SENTIMENT = True
SHOW_INTELLIGENCE = True
SHOW_ASYMMETRY = True
SHOW_EMOJIS = True
SHOW_TOP_WORDS = True
SHOW_EMOTIONAL_ANALYSIS = True

PLOT_PIE = True
PLOT_HOURLY_ACTIVITY = True
PLOT_MOOD_TREND = True
PLOT_HEATMAP = True
PLOT_TIMELINE = True
PLOT_REPLY_DIST = True
PLOT_WORDCLOUD = True
PLOT_WORDCLOUD_PER_PERSON = True
PLOT_EMOTIONAL_TIMELINE = True
PLOT_DATE_ANALYSIS = True

SHOW_PROGRESS = True
FAST_MODE = False

CONVO_GAP = 60
MAX_SENDERS = None

SYSTEM_MESSAGE_KEYWORDS = [
    "end-to-end",
    "disappearing messages",
    "created this group",
    "added",
    "changed",
    "left",
    "removed",
    "joined using",
]

WORDCLOUD_PERSON_COLORS = [
    "cool",
    "autumn",
    "winter",
    "spring",
    "summer",
    "plasma",
    "cividis",
    "magma",
    "viridis",
]

POSITIVE_EMOJIS = {
    "😀",
    "😄",
    "😁",
    "😆",
    "😊",
    "😍",
    "🥰",
    "😘",
    "❤️",
    "❤",
    "💖",
    "💯",
    "🎉",
    "👏",
    "🙏",
    "🔥",
    "✨",
    "😎",
    "👍",
}

NEGATIVE_EMOJIS = {
    "😢",
    "😭",
    "😞",
    "😔",
    "😡",
    "😠",
    "🤬",
    "💔",
    "😣",
    "😩",
    "😫",
    "😤",
    "👎",
}

DEFAULT_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "is",
    "it",
    "for",
    "on",
    "at",
    "this",
    "that",
    "i",
    "you",
    "we",
    "they",
    "me",
    "my",
    "your",
    "our",
    "their",
    "be",
    "are",
    "was",
    "were",
    "with",
    "as",
    "by",
    "from",
    "but",
    "so",
    "if",
    "then",
    "not",
    "omitted",
    "media",
    "image",
    "video",
    "sticker",
}

plt.style.use("dark_background")


def progress(message: str) -> None:
    if SHOW_PROGRESS:
        print(f"[progress] {message}")


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    mins, sec = divmod(seconds, 60)
    if mins < 60:
        return f"{mins}m {sec}s" if sec else f"{mins}m"
    hours, mins = divmod(mins, 60)
    if mins and sec:
        return f"{hours}h {mins}m {sec}s"
    if mins:
        return f"{hours}h {mins}m"
    if sec:
        return f"{hours}h {sec}s"
    return f"{hours}h"


def sanitize_sender(sender: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", sender.strip())
    return cleaned.strip("_") or "unknown"


def maybe_fix_mojibake(text: str) -> str:
    def score(candidate: str) -> int:
        return sum(candidate.count(ch) for ch in ("Ã", "â", "ð", "�"))

    try:
        repaired = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return text
    return repaired if repaired and score(repaired) < score(text) else text


def clean_text(text: str) -> str:
    cleaned = text.replace("\ufeff", "").replace("\u202f", " ").replace("\u200e", "")
    if ftfy is not None:
        cleaned = ftfy.fix_text(cleaned)
    return maybe_fix_mojibake(cleaned)


def parse_timestamp(ts: str) -> Optional[pd.Timestamp]:
    ts = ts.strip().replace("  ", " ")
    formats = [
        "%d/%m/%y, %I:%M:%S %p",
        "%d/%m/%Y, %I:%M:%S %p",
        "%d/%m/%y, %I:%M %p",
        "%d/%m/%Y, %I:%M %p",
        "%d/%m/%y, %H:%M:%S",
        "%d/%m/%Y, %H:%M:%S",
        "%d/%m/%y, %H:%M",
        "%d/%m/%Y, %H:%M",
        "%m/%d/%y, %I:%M:%S %p",
        "%m/%d/%Y, %I:%M:%S %p",
        "%m/%d/%y, %I:%M %p",
        "%m/%d/%Y, %I:%M %p",
    ]
    for fmt in formats:
        parsed = pd.to_datetime(ts, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return parsed
    fallback = pd.to_datetime(ts, errors="coerce", dayfirst=True)
    return fallback if pd.notna(fallback) else None


def parse_chat(raw_text: str) -> pd.DataFrame:
    line_pattern = re.compile(r"^\[(?P<ts>[^\]]+)\]\s(?P<body>.*)$")
    sender_pattern = re.compile(r"^(?P<sender>[^:]+):\s?(?P<message>.*)$")

    rows: List[Dict[str, object]] = []
    last_index: Optional[int] = None

    for raw_line in raw_text.splitlines():
        line = clean_text(raw_line.rstrip("\n"))
        if not line.strip():
            continue

        line_match = line_pattern.match(line)
        if line_match:
            ts = parse_timestamp(line_match.group("ts"))
            body = line_match.group("body").strip()
            sender_match = sender_pattern.match(body)
            if ts is None or sender_match is None:
                last_index = None
                continue

            sender = sender_match.group("sender").strip()
            message = sender_match.group("message").strip()
            lower_message = message.lower()
            if any(keyword in lower_message for keyword in SYSTEM_MESSAGE_KEYWORDS):
                last_index = None
                continue

            rows.append({"DateTime": ts, "Sender": sender, "Message": message})
            last_index = len(rows) - 1
        elif last_index is not None:
            continuation = clean_text(line)
            if continuation:
                rows[last_index]["Message"] = f"{rows[last_index]['Message']}\n{continuation}"

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("DateTime").reset_index(drop=True)
    df["Sender"] = df["Sender"].astype(str)

    if MAX_SENDERS is not None:
        allowed = df["Sender"].value_counts().head(MAX_SENDERS).index.tolist()
        df = df[df["Sender"].isin(allowed)].copy().reset_index(drop=True)
    return df


def get_stopwords() -> set:
    try:
        import nltk
        from nltk.corpus import stopwords

        nltk.download("stopwords", quiet=True)
        words = set(stopwords.words("english"))
        words.update(DEFAULT_STOPWORDS)
        return words
    except Exception:
        return set(DEFAULT_STOPWORDS)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text.lower())


def emoji_boost(text: str) -> float:
    score = 0.0
    for ch in text:
        if ch in POSITIVE_EMOJIS:
            score += 0.12
        elif ch in NEGATIVE_EMOJIS:
            score -= 0.12
    return max(-0.35, min(0.35, score))


def sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def sentiment_for_message(text: str, analyzer: SentimentIntensityAnalyzer) -> float:
    words = tokenize(text)
    if len(words) > 10:
        sentences = sentence_split(text)
        if len(sentences) > 1:
            scores = []
            for sentence in sentences:
                s = analyzer.polarity_scores(sentence)["compound"] + emoji_boost(sentence)
                scores.append(max(-1.0, min(1.0, s)))
            if scores:
                return max(scores, key=lambda x: abs(x))
    score = analyzer.polarity_scores(text)["compound"] + emoji_boost(text)
    return max(-1.0, min(1.0, score))


def lexical_metrics(messages: Iterable[str], stop_words: set) -> Dict[str, float]:
    tokens: List[str] = []
    for msg in messages:
        tokens.extend(tokenize(msg))

    if not tokens:
        return {
            "total_words": 0.0,
            "unique_words": 0.0,
            "ttr": 0.0,
            "lexical_density": 0.0,
            "avg_word_len": 0.0,
            "vocab_score": 0.0,
            "complexity_score": 0.0,
        }

    total_words = len(tokens)
    unique_words = len(set(tokens))
    content_words = [w for w in tokens if w not in stop_words and len(w) > 2]
    avg_word_len = mean(len(w) for w in tokens)
    long_ratio = sum(1 for w in tokens if len(w) >= 7) / total_words

    ttr = unique_words / total_words
    lexical_density = len(content_words) / total_words
    vocab_score = clamp((ttr / 0.60) * 60 + lexical_density * 30 + (min(avg_word_len, 8.0) / 8.0) * 10)
    complexity = clamp(lexical_density * 40 + long_ratio * 30 + (min(avg_word_len, 8.0) / 8.0) * 30)

    return {
        "total_words": float(total_words),
        "unique_words": float(unique_words),
        "ttr": ttr,
        "lexical_density": lexical_density,
        "avg_word_len": avg_word_len,
        "vocab_score": vocab_score,
        "complexity_score": complexity,
    }


def sentiment_distribution(scores: List[float]) -> Dict[str, float | str]:
    if not scores:
        return {
            "pct_positive": 0.0,
            "pct_neutral": 0.0,
            "pct_negative": 0.0,
            "weighted_positivity": 0.0,
            "raw_average": 0.0,
            "grade": "Mostly Neutral 😐",
        }

    total = len(scores)
    positive = sum(1 for s in scores if s > 0.05)
    neutral = sum(1 for s in scores if -0.05 <= s <= 0.05)
    negative = sum(1 for s in scores if s < -0.05)

    pct_positive = positive / total
    pct_neutral = neutral / total
    pct_negative = negative / total
    weighted = pct_positive - pct_negative
    raw_average = mean(scores)

    if weighted >= 0.5:
        grade = "Very Positive 🌟"
    elif weighted >= 0.2:
        grade = "Mildly Positive 😊"
    elif weighted >= -0.2:
        grade = "Mostly Neutral 😐"
    elif weighted >= -0.5:
        grade = "Mildly Negative 😕"
    else:
        grade = "Very Negative 😞"

    return {
        "pct_positive": pct_positive,
        "pct_neutral": pct_neutral,
        "pct_negative": pct_negative,
        "weighted_positivity": weighted,
        "raw_average": raw_average,
        "grade": grade,
    }


def fig_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def file_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def make_wordcloud(text: str, output_path: Path, colormap: str) -> Optional[str]:
    if not text.strip():
        return None
    max_words = 150 if FAST_MODE else 300
    wc = WordCloud(
        width=1400,
        height=800,
        background_color="black",
        collocations=False,
        max_words=max_words,
        colormap=colormap,
    ).generate(text)
    wc.to_file(str(output_path))
    return file_to_base64(output_path)


def build_charts(
    df: pd.DataFrame,
    senders: List[str],
    reply_times_by_sender: Dict[str, List[float]],
    reply_scatter_rows: List[Dict[str, object]],
) -> Dict[str, str]:
    charts: Dict[str, str] = {}

    if PLOT_PIE:
        progress("Building pie chart...")
        fig, ax = plt.subplots(figsize=(8, 6))
        counts = df["Sender"].value_counts().reindex(senders)
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
        ax.set_title("Message Share by Sender")
        charts["pie"] = fig_to_base64(fig)

    if PLOT_HOURLY_ACTIVITY:
        progress("Building hourly activity chart...")
        fig, ax = plt.subplots(figsize=(10, 5))
        for sender in senders:
            subset = df[df["Sender"] == sender]
            hourly = subset["DateTime"].dt.hour.value_counts().sort_index().reindex(range(24), fill_value=0)
            ax.plot(hourly.index, hourly.values, marker="o", label=sender)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Messages")
        ax.set_title("Hourly Activity")
        ax.legend(loc="upper right", fontsize=8)
        charts["hourly"] = fig_to_base64(fig)

    if PLOT_MOOD_TREND:
        progress("Building mood trend chart...")
        fig, ax = plt.subplots(figsize=(10, 5))
        date_df = df.set_index("DateTime")
        for sender in senders:
            series = date_df[date_df["Sender"] == sender]["Sentiment"].resample("D").mean()
            if not series.empty:
                ax.plot(series.index, series.values, marker="o", label=sender)
        ax.axhline(0, color="#888", linestyle="--", linewidth=1)
        ax.set_title("Daily Mood Trend (Sentiment)")
        ax.set_ylabel("Sentiment")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        fig.autofmt_xdate()
        ax.legend(loc="best", fontsize=8)
        charts["mood"] = fig_to_base64(fig)

    if PLOT_HEATMAP:
        progress("Building heatmap...")
        fig, ax = plt.subplots(figsize=(12, 5))
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        temp = df.copy()
        temp["dow"] = temp["DateTime"].dt.dayofweek
        temp["hour"] = temp["DateTime"].dt.hour
        heat = (
            temp.pivot_table(index="dow", columns="hour", values="Message", aggfunc="count", fill_value=0)
            .reindex(index=range(7), columns=range(24), fill_value=0)
        )
        image = ax.imshow(heat.values, aspect="auto", cmap="inferno")
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        ax.set_yticks(range(7))
        ax.set_yticklabels(day_names)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels(range(0, 24, 2))
        ax.set_title("Activity Heatmap")
        ax.set_xlabel("Hour")

        row_totals = heat.sum(axis=1).tolist()
        for i, total in enumerate(row_totals):
            ax.text(24.2, i, str(total), va="center", ha="left", color="white", fontsize=9)
        ax.text(0.99, 1.02, f"Total messages: {len(df)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
        ax.text(1.01, 1.02, "Row total", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="#bbb")
        ax.set_xlim(-0.5, 25.0)
        charts["heatmap"] = fig_to_base64(fig)

    if PLOT_TIMELINE:
        progress("Building timeline chart...")
        fig, ax = plt.subplots(figsize=(11, 5))
        for sender in senders:
            subset = df[df["Sender"] == sender].copy()
            subset["CumMessages"] = range(1, len(subset) + 1)
            ax.plot(subset["DateTime"], subset["CumMessages"], label=sender)
        ax.set_title("Cumulative Messages Over Time")
        ax.set_ylabel("Cumulative Messages")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        fig.autofmt_xdate()
        ax.legend(loc="best", fontsize=8)
        charts["timeline"] = fig_to_base64(fig)

    if PLOT_REPLY_DIST:
        progress("Building reply distribution chart...")
        all_reply_times = [v for values in reply_times_by_sender.values() for v in values]
        if all_reply_times:
            clip_value = pd.Series(all_reply_times).quantile(0.95)
            clipped = [min(v, clip_value) for v in all_reply_times]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(clipped, bins=30, density=True, alpha=0.9, color="#33b5e5")
            ax.set_title("Reply Time Distribution (seconds, clipped at 95th percentile)")
            ax.set_xlabel("Reply time (seconds)")
            ax.set_ylabel("Density")
            charts["reply_dist"] = fig_to_base64(fig)

    if PLOT_EMOTIONAL_TIMELINE:
        progress("Building emotional timeline chart...")
        fig, ax = plt.subplots(figsize=(11, 4.5))
        rolling = df["IsEmotional"].astype(int).rolling(window=20 if not FAST_MODE else 10, min_periods=1).sum()
        ax.plot(df["DateTime"], rolling, color="#ff6b6b", linewidth=1.8)
        ax.set_title("Emotional Message Timeline (rolling count)")
        ax.set_ylabel("Rolling emotional count")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        fig.autofmt_xdate()
        charts["emotional_timeline"] = fig_to_base64(fig)

    if PLOT_DATE_ANALYSIS:
        progress("Building date analysis charts...")
        date_df = df.set_index("DateTime")

        daily = date_df.resample("D").size()
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(daily.index, daily.values, marker="o", linewidth=1.5)
        ax.set_title("Messages per Day")
        ax.set_ylabel("Messages")
        ax.set_xlim(df["DateTime"].min(), df["DateTime"].max())
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        fig.autofmt_xdate()
        charts["messages_per_day"] = fig_to_base64(fig)

        weekly = date_df.resample("W").size()
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(weekly.index, weekly.values, marker="o", linewidth=1.5, color="#4dd0e1")
        ax.set_title("Messages per Week")
        ax.set_ylabel("Messages")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        fig.autofmt_xdate()
        charts["messages_per_week"] = fig_to_base64(fig)

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_counts = df["DateTime"].dt.day_name().value_counts().reindex(day_order, fill_value=0)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.bar(dow_counts.index, dow_counts.values, color="#ab47bc")
        ax.set_title("Day-of-Week Distribution")
        ax.set_ylabel("Messages")
        ax.tick_params(axis="x", rotation=30)
        charts["day_of_week"] = fig_to_base64(fig)

        scatter_df = pd.DataFrame(reply_scatter_rows)
        if not scatter_df.empty:
            fig, ax = plt.subplots(figsize=(11, 4.8))
            for sender in senders:
                subset = scatter_df[scatter_df["Sender"] == sender]
                if not subset.empty:
                    ax.scatter(subset["DateTime"], subset["ReplySeconds"], s=20, alpha=0.7, label=sender)
            ax.set_title("Response Time Over Calendar Time")
            ax.set_ylabel("Reply time (seconds)")
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            fig.autofmt_xdate()
            ax.legend(loc="best", fontsize=8)
            charts["response_time_over_time"] = fig_to_base64(fig)

    return charts


def build_emotional_clusters(df: pd.DataFrame) -> List[Dict[str, object]]:
    clusters: List[Dict[str, object]] = []
    for _, convo in df.groupby("ConversationID", sort=True):
        convo = convo.reset_index(drop=True)
        i = 0
        while i < len(convo):
            if not bool(convo.loc[i, "IsEmotional"]):
                i += 1
                continue
            start = i
            while i < len(convo) and bool(convo.loc[i, "IsEmotional"]):
                i += 1
            end = i - 1
            cluster_rows = convo.loc[start:end]
            context = convo.loc[start - 1, "Message"] if start > 0 else "(Conversation started here)"
            clusters.append(
                {
                    "start": cluster_rows.iloc[0]["DateTime"],
                    "senders": ", ".join(cluster_rows["Sender"].value_counts().index.tolist()),
                    "length": len(cluster_rows),
                    "avg_sentiment": cluster_rows["Sentiment"].mean(),
                    "context": str(context)[:180],
                }
            )

    clusters.sort(key=lambda x: (abs(float(x["avg_sentiment"])), int(x["length"])), reverse=True)
    return clusters[:10]


def html_card(title: str, body: str) -> str:
    return f"<div class='card'><h3>{title}</h3>{body}</div>"


def make_dashboard(
    output_path: Path,
    df: pd.DataFrame,
    senders: List[str],
    lexical_by_sender: Dict[str, Dict[str, float]],
    sentiment_by_sender: Dict[str, Dict[str, float | str]],
    intelligence_by_sender: Dict[str, Dict[str, float]],
    reply_times_by_sender: Dict[str, List[float]],
    conversation_stats: Dict[str, Dict[str, int]],
    top_words: List[tuple],
    top_emojis: List[tuple],
    emotional_stats: Dict[str, Dict[str, float]],
    emotional_clusters: List[Dict[str, object]],
    charts: Dict[str, str],
    wordcloud_global_b64: Optional[str],
    wordcloud_person_b64: Dict[str, Optional[str]],
) -> None:
    progress("Building HTML dashboard...")

    total_conversations = int(df["ConversationID"].nunique()) if not df.empty else 0
    date_start = df["DateTime"].min().strftime("%d %b %Y") if not df.empty else "N/A"
    date_end = df["DateTime"].max().strftime("%d %b %Y") if not df.empty else "N/A"

    global_bar = f"""
    <div class='global-bar'>
      <div><strong>{len(df):,}</strong><span>Messages</span></div>
      <div><strong>{len(senders)}</strong><span>Senders</span></div>
      <div><strong>{total_conversations}</strong><span>Conversations</span></div>
      <div><strong>{date_start} → {date_end}</strong><span>Date range</span></div>
    </div>
    """

    cards: List[str] = []
    if SHOW_TOP_WORDS:
        word_html = "<ol>" + "".join(f"<li><code>{html.escape(w)}</code> — {c}</li>" for w, c in top_words) + "</ol>"
        cards.append(html_card("Top Meaningful Words", word_html))

    if SHOW_EMOJIS:
        emoji_html = "<ol>" + "".join(f"<li>{html.escape(e)} — {c}</li>" for e, c in top_emojis) + "</ol>"
        cards.append(html_card("Top Emojis", emoji_html))

    for sender in senders:
        sections = [f"<p><strong>Messages:</strong> {int((df['Sender'] == sender).sum())}</p>"]
        if SHOW_BASIC_STATS:
            cstats = conversation_stats.get(sender, {})
            sections.append(
                "<ul>"
                f"<li><strong>Conversation starters:</strong> {cstats.get('starters', 0)}</li>"
                f"<li><strong>Conversation enders:</strong> {cstats.get('enders', 0)}</li>"
                f"<li><strong>Conversation revivers:</strong> {cstats.get('revivers', 0)}</li>"
                "</ul>"
            )

        if SHOW_REPLY_TIMES:
            values = reply_times_by_sender.get(sender, [])
            if values:
                sections.append(
                    f"<p><strong>Reply times:</strong> avg {format_duration(mean(values))}, "
                    f"median {format_duration(median(values))}, max {format_duration(max(values))}</p>"
                )
            else:
                sections.append("<p><strong>Reply times:</strong> Not enough sender-switch replies.</p>")

        if SHOW_LEXICAL:
            lex = lexical_by_sender.get(sender, {})
            sections.append(
                f"<p><strong>Lexical:</strong> TTR {lex.get('ttr', 0.0):.2f}, "
                f"density {lex.get('lexical_density', 0.0):.2f}, "
                f"avg word len {lex.get('avg_word_len', 0.0):.2f}</p>"
            )
            sections.append(
                f"<p><strong>Vocabulary score:</strong> {lex.get('vocab_score', 0.0):.1f}/100 "
                f"&nbsp;|&nbsp; <strong>Language complexity:</strong> {lex.get('complexity_score', 0.0):.1f}/100</p>"
            )

        if SHOW_SENTIMENT:
            sent = sentiment_by_sender.get(sender, {})
            sections.append(
                f"<p><strong>Weighted positivity score:</strong> {float(sent.get('weighted_positivity', 0.0)):.2f}</p>"
                f"<p><strong>Positive:</strong> {float(sent.get('pct_positive', 0.0)) * 100:.1f}% &nbsp;"
                f"<strong>Neutral:</strong> {float(sent.get('pct_neutral', 0.0)) * 100:.1f}% &nbsp;"
                f"<strong>Negative:</strong> {float(sent.get('pct_negative', 0.0)) * 100:.1f}%</p>"
                f"<p><strong>Sentiment grade:</strong> {html.escape(str(sent.get('grade', 'Mostly Neutral 😐')))}</p>"
                f"<p><strong>Raw VADER average (reference):</strong> {float(sent.get('raw_average', 0.0)):.2f}</p>"
                "<p class='note'>Distribution + weighted positivity is more accurate than a plain average "
                "because many chat lines are naturally neutral.</p>"
            )

        if SHOW_INTELLIGENCE:
            intel = intelligence_by_sender.get(sender, {})
            sections.append(
                f"<p><strong>Conversation Intelligence:</strong> {intel.get('ci_score', 0.0):.1f}/100 "
                f"(engagement {intel.get('engagement_score', 0.0):.1f}, "
                f"lexical {intel.get('lexical_score', 0.0):.1f}, "
                f"complexity {intel.get('complexity_score', 0.0):.1f})</p>"
            )

        if SHOW_EMOTIONAL_ANALYSIS:
            emos = emotional_stats.get(sender, {})
            sections.append(
                f"<p><strong>Highly emotional messages:</strong> {emos.get('pct_emotional', 0.0) * 100:.1f}%</p>"
                f"<p><strong>Positive-emotional:</strong> {emos.get('pct_pos_emotional', 0.0) * 100:.1f}% &nbsp;"
                f"<strong>Negative-emotional:</strong> {emos.get('pct_neg_emotional', 0.0) * 100:.1f}%</p>"
            )

        cards.append(html_card(html.escape(sender), "".join(sections)))

    chart_titles = {
        "pie": "Message Share",
        "hourly": "Hourly Activity",
        "mood": "Mood Trend",
        "heatmap": "Activity Heatmap",
        "timeline": "Cumulative Timeline",
        "reply_dist": "Reply Time Distribution",
        "emotional_timeline": "Emotional Timeline",
        "messages_per_day": "Messages per Day",
        "messages_per_week": "Messages per Week",
        "day_of_week": "Day-of-Week Distribution",
        "response_time_over_time": "Response Time Over Calendar Time",
    }
    chart_blocks = []
    for key, title in chart_titles.items():
        if key in charts:
            chart_blocks.append(
                f"<div class='chart-card'><h3>{title}</h3><img src='data:image/png;base64,{charts[key]}' alt='{title}'></div>"
            )

    wc_blocks = []
    if PLOT_WORDCLOUD and wordcloud_global_b64:
        wc_blocks.append(
            "<div class='chart-card'><h3>Global Word Cloud</h3>"
            f"<img src='data:image/png;base64,{wordcloud_global_b64}' alt='Global word cloud'></div>"
        )
    if PLOT_WORDCLOUD_PER_PERSON:
        for sender in senders:
            wc = wordcloud_person_b64.get(sender)
            if wc:
                wc_blocks.append(
                    f"<div class='chart-card'><h3>{html.escape(sender)} Word Cloud</h3>"
                    f"<img src='data:image/png;base64,{wc}' alt='{html.escape(sender)} word cloud'></div>"
                )

    if emotional_clusters:
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(pd.to_datetime(c['start']).strftime('%d %b %Y %H:%M'))}</td>"
            f"<td>{html.escape(str(c['senders']))}</td>"
            f"<td>{int(c['length'])}</td>"
            f"<td>{float(c['avg_sentiment']):.2f}</td>"
            f"<td>{html.escape(str(c['context']))}</td>"
            "</tr>"
            for c in emotional_clusters
        )
        emotional_section = (
            "<details class='details-block'><summary>Top Emotional Clusters (click to expand)</summary>"
            "<table><thead><tr><th>Start</th><th>Senders</th><th>Length</th><th>Avg sentiment</th><th>Context message</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></details>"
        )
    else:
        emotional_section = (
            "<details class='details-block'><summary>Top Emotional Clusters</summary>"
            "<p>No emotional clusters detected.</p></details>"
        )

    score_explainer = """
    <section class='score-explainer'>
      <h2>Score Explainer</h2>
      <ul>
        <li><strong>Vocabulary score:</strong> type-token ratio + lexical density + average word length.</li>
        <li><strong>Language complexity:</strong> lexical density + long-word ratio + average word length.</li>
        <li><strong>Conversation Intelligence (CI):</strong> 50% engagement + 25% lexical + 25% complexity.</li>
        <li><strong>Sentiment:</strong> weighted positivity from positive/neutral/negative percentages with raw VADER as reference.</li>
      </ul>
    </section>
    """

    html_doc = f"""
    <!doctype html>
    <html lang='en'>
    <head>
      <meta charset='utf-8'>
      <meta name='viewport' content='width=device-width, initial-scale=1'>
      <title>Ultimate Conversation Analysis</title>
      <style>
        body {{ background: #0d0d0d; color: #f2f2f2; font-family: Inter, Segoe UI, Arial, sans-serif; margin: 0; padding: 24px; }}
        h1 {{ margin: 0 0 18px; font-size: 2rem; background: linear-gradient(90deg, #7f5af0, #2cb67d); -webkit-background-clip: text; background-clip: text; color: transparent; }}
        h2 {{ margin-top: 30px; }}
        h3 {{ margin-top: 0; }}
        .global-bar {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin-bottom: 20px; }}
        .global-bar > div {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px; padding: 12px; }}
        .global-bar strong {{ display: block; font-size: 1.15rem; }}
        .global-bar span {{ color: #b0b0b0; font-size: 0.9rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
        .card, .chart-card {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; padding: 14px; }}
        .chart-card img {{ width: 100%; border-radius: 8px; }}
        .details-block {{ margin-top: 22px; background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; padding: 12px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.93rem; }}
        th, td {{ border: 1px solid #2a2a2a; padding: 8px; text-align: left; vertical-align: top; }}
        th {{ background: #111; }}
        .note {{ color: #b8b8b8; font-size: 0.9rem; }}
        .score-explainer {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; padding: 16px; margin-top: 28px; }}
      </style>
    </head>
    <body>
      <h1>Ultimate Conversation Analysis</h1>
      {global_bar}
      <h2>Stats</h2>
      <section class='grid'>{''.join(cards)}</section>
      <h2>Charts</h2>
      <section class='grid'>{''.join(chart_blocks)}</section>
      <h2>Word Clouds</h2>
      <section class='grid'>{''.join(wc_blocks)}</section>
      {emotional_section}
      {score_explainer}
    </body>
    </html>
    """
    output_path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "chat_input.txt"
    output_path = base_dir / "ultimate_analysis.html"

    progress("Starting analysis...")
    if not input_path.exists():
        print(f"chat_input.txt not found at {input_path}")
        return

    raw = input_path.read_text(encoding="utf-8", errors="replace")
    progress("Parsing chat messages...")
    df = parse_chat(raw)
    if df.empty:
        print("No valid chat messages found after parsing.")
        return

    senders = df["Sender"].value_counts().index.tolist()
    df["PrevDateTime"] = df["DateTime"].shift(1)
    df["GapMinutes"] = (df["DateTime"] - df["PrevDateTime"]).dt.total_seconds().div(60)
    df["ConversationStart"] = df["PrevDateTime"].isna() | (df["GapMinutes"] >= CONVO_GAP)
    df["ConversationID"] = df["ConversationStart"].cumsum().astype(int)

    progress("Scoring sentiment...")
    analyzer = SentimentIntensityAnalyzer()
    df["Sentiment"] = df["Message"].apply(lambda msg: sentiment_for_message(str(msg), analyzer))
    df["IsEmotional"] = df["Sentiment"].abs() > 0.5

    progress("Computing conversation and reply metrics...")
    starters = df.groupby("ConversationID").first()["Sender"].value_counts().to_dict()
    enders = df.groupby("ConversationID").last()["Sender"].value_counts().to_dict()
    revivers = df[df["ConversationStart"] & (df["ConversationID"] > 1)]["Sender"].value_counts().to_dict()
    conversation_stats = {
        sender: {
            "starters": int(starters.get(sender, 0)),
            "enders": int(enders.get(sender, 0)),
            "revivers": int(revivers.get(sender, 0)),
        }
        for sender in senders
    }

    reply_times_by_sender: Dict[str, List[float]] = defaultdict(list)
    reply_scatter_rows: List[Dict[str, object]] = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        if int(row["ConversationID"]) != int(prev["ConversationID"]):
            continue
        if row["Sender"] == prev["Sender"]:
            continue
        gap_seconds = (row["DateTime"] - prev["DateTime"]).total_seconds()
        if gap_seconds < 0 or gap_seconds >= CONVO_GAP * 60:
            continue
        sender = str(row["Sender"])
        reply_times_by_sender[sender].append(float(gap_seconds))
        reply_scatter_rows.append({"Sender": sender, "DateTime": row["DateTime"], "ReplySeconds": float(gap_seconds)})

    progress("Computing lexical/sentiment/intelligence metrics...")
    stop_words = get_stopwords()
    lexical_by_sender = {
        sender: lexical_metrics(df[df["Sender"] == sender]["Message"].tolist(), stop_words) for sender in senders
    }
    sentiment_by_sender = {
        sender: sentiment_distribution(df[df["Sender"] == sender]["Sentiment"].tolist()) for sender in senders
    }

    convo_count = max(int(df["ConversationID"].nunique()), 1)
    intelligence_by_sender: Dict[str, Dict[str, float]] = {}
    for sender in senders:
        count = int((df["Sender"] == sender).sum())
        message_share = count / len(df)
        starter_share = conversation_stats[sender]["starters"] / convo_count
        reviver_share = conversation_stats[sender]["revivers"] / max(convo_count - 1, 1)
        rtimes = reply_times_by_sender.get(sender, [])
        reply_factor = 1.0 - min(median(rtimes), 3600.0) / 3600.0 if rtimes else 0.5
        engagement = clamp(message_share * 45 + starter_share * 20 + reviver_share * 15 + reply_factor * 20)
        lexical_score = lexical_by_sender[sender]["vocab_score"]
        complexity_score = lexical_by_sender[sender]["complexity_score"]
        ci = clamp(engagement * 0.50 + lexical_score * 0.25 + complexity_score * 0.25)
        intelligence_by_sender[sender] = {
            "engagement_score": engagement,
            "lexical_score": lexical_score,
            "complexity_score": complexity_score,
            "ci_score": ci,
        }

    progress("Computing top words and emojis...")
    all_tokens = []
    for msg in df["Message"]:
        all_tokens.extend([w for w in tokenize(str(msg)) if w not in stop_words and len(w) > 2])
    top_words = Counter(all_tokens).most_common(20)
    all_emojis = []
    for msg in df["Message"].astype(str).tolist():
        all_emojis.extend([item["emoji"] for item in emoji.emoji_list(msg)])
    top_emojis = Counter(all_emojis).most_common(20)

    progress("Computing emotional analysis...")
    emotional_stats: Dict[str, Dict[str, float]] = {}
    for sender in senders:
        subset = df[df["Sender"] == sender]
        total = len(subset)
        emotional_subset = subset[subset["IsEmotional"]]
        emotional_count = len(emotional_subset)
        pos_emotional = len(emotional_subset[emotional_subset["Sentiment"] > 0])
        neg_emotional = len(emotional_subset[emotional_subset["Sentiment"] < 0])
        emotional_stats[sender] = {
            "pct_emotional": emotional_count / total if total else 0.0,
            "pct_pos_emotional": pos_emotional / emotional_count if emotional_count else 0.0,
            "pct_neg_emotional": neg_emotional / emotional_count if emotional_count else 0.0,
        }
    emotional_clusters = build_emotional_clusters(df)

    charts = build_charts(df, senders, reply_times_by_sender, reply_scatter_rows)

    progress("Generating word clouds...")
    wordcloud_global_b64 = None
    wordcloud_person_b64: Dict[str, Optional[str]] = {}
    if PLOT_WORDCLOUD:
        global_wc_path = base_dir / "wordcloud_global.png"
        wordcloud_global_b64 = make_wordcloud(" ".join(df["Message"].astype(str).tolist()), global_wc_path, "viridis")

    if PLOT_WORDCLOUD_PER_PERSON:
        for i, sender in enumerate(senders):
            filename = f"wordcloud_{sanitize_sender(sender)}.png"
            cmap = WORDCLOUD_PERSON_COLORS[i % len(WORDCLOUD_PERSON_COLORS)]
            sender_text = " ".join(df[df["Sender"] == sender]["Message"].astype(str).tolist())
            wordcloud_person_b64[sender] = make_wordcloud(sender_text, base_dir / filename, cmap)

    make_dashboard(
        output_path=output_path,
        df=df,
        senders=senders,
        lexical_by_sender=lexical_by_sender,
        sentiment_by_sender=sentiment_by_sender,
        intelligence_by_sender=intelligence_by_sender,
        reply_times_by_sender=reply_times_by_sender,
        conversation_stats=conversation_stats,
        top_words=top_words,
        top_emojis=top_emojis,
        emotional_stats=emotional_stats,
        emotional_clusters=emotional_clusters,
        charts=charts,
        wordcloud_global_b64=wordcloud_global_b64,
        wordcloud_person_b64=wordcloud_person_b64,
    )
    progress(f"Done. HTML written to: {output_path}")


if __name__ == "__main__":
    main()
