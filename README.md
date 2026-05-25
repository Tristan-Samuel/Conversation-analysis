# Ultimate WhatsApp Conversation Analysis

A full-featured WhatsApp chat analyzer that reads exported chat text and produces a dark-themed HTML dashboard with stats, sentiment insights, emotional analysis, lexical scoring, and charts.

## Features

- Parses WhatsApp chats with flexible timestamp formats
- Supports multi-line messages
- Handles encoding cleanup (BOM, mojibake cleanup, optional `ftfy`)
- Supports any number of senders
- Conversation segmentation with starter/ender/reviver stats
- Reply-time analysis in seconds (avg, median, max + density distribution chart)
- Sentiment analysis with:
  - distribution (% positive / neutral / negative)
  - weighted positivity score
  - sentence-level peak sentiment for long messages
  - sentiment grade
- Emotional analysis:
  - highly emotional message rate
  - positive vs negative emotional share
  - emotional timeline
  - top emotional clusters with context trigger
- Lexical analysis:
  - type-token ratio
  - lexical density
  - average word length
  - vocabulary score (/100)
  - language complexity score (/100)
- Conversation Intelligence (CI) scoring
- Global + per-sender word clouds
- Date-based analysis charts:
  - messages/day (fixed date axis)
  - messages/week
  - day-of-week distribution
  - response time over calendar time
- Fully embedded HTML output (base64 charts and word clouds)
- Feature toggles at the top of `analysis.py`

## Installation

```bash
pip install pandas nltk vaderSentiment wordcloud matplotlib emoji ftfy
```

## Usage

1. Export your WhatsApp chat as a `.txt` file (without media).
2. Place it in this project directory as:
   - `chat_input.txt`
3. Run:

```bash
python analysis.py
```

The script reads `chat_input.txt` and outputs:

- `ultimate_analysis.html`
- `wordcloud_global.png`
- `wordcloud_<SenderName>.png` for each sender

## How to Export WhatsApp Chats

On WhatsApp:

**Settings → Chats → Export Chat → Without Media**

Then copy the exported `.txt` content into `chat_input.txt`.

## Score Explanations

- **Vocabulary score (/100)**  
  Heuristic score using type-token ratio, lexical density, and average word length.

- **Language complexity (/100)**  
  Heuristic score using lexical density, long-word ratio, and average word length.

- **Conversation Intelligence (CI) (/100)**  
  Composite score:
  - 50% engagement score
  - 25% lexical component
  - 25% language complexity component

- **Sentiment score**  
  Uses sentiment distribution (% positive/neutral/negative), weighted positivity, and raw VADER average (reference only).  
  Weighted positivity is more representative than plain averages because many chat lines are naturally neutral.

## Limitations / Disclaimers

- This is heuristic analysis, not a psychological or clinical assessment.
- Sentiment and intelligence-style scores are approximate and context-dependent.
- Sarcasm, slang, code-switching, and domain-specific language may reduce sentiment accuracy.
- WhatsApp export formats vary by locale/device; uncommon formats may need additional parsing rules.

## License

MIT