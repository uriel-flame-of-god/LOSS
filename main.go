package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"
	"unicode"
)

type Config struct {
	StripMarkdown    bool
	StripEmojis      bool
	NormalizePunct   bool
	RemoveLLMPhrases bool
	FlattenVocab     string
	RephraseRatio    float64
	Domain           string // Reserved for future use
	UseLocalModel    bool
	APIEndpoint      string
	APIKey           string
	ModelName        string
	Temperature      float64
	Seed             int64
}

type Sanitizer struct {
	config Config
	rng    *rand.Rand
	client *http.Client
}

// OpenAI-compatible API request/response structures
// NOTE: This is NOT a full OpenAI client - simplified for local model use
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
}

type ChatCompletionResponse struct {
	Choices []struct {
		Message ChatMessage `json:"message"`
	} `json:"choices"`
}

// Sentence with preserved punctuation
type Sentence struct {
	Text        string
	Punctuation string
}

func NewSanitizer(cfg Config) *Sanitizer {
	seed := cfg.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	return &Sanitizer{
		config: cfg,
		rng:    rand.New(rand.NewSource(seed)),
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (s *Sanitizer) Sanitize(text string) string {
	text = s.stage1_structuralNormalization(text)
	text = s.stage2_punctuationNormalization(text)
	text = s.stage3_emojiRemoval(text)
	text = s.stage5_sentenceRhythmDeoptimization(text)
	text = s.stage6_vocabularyFlattening(text)
	text = s.stage7_partialRephrasing(text)
	text = s.stage4_llmPhraseSuppression(text) // Run AFTER rephrasing to catch reintroduced phrases
	text = s.stage8_finalPass(text)
	return text
}

// Stage 1: Structural Normalization
func (s *Sanitizer) stage1_structuralNormalization(text string) string {
	if !s.config.StripMarkdown {
		return text
	}

	// Remove code blocks (non-greedy)
	codeBlockRe := regexp.MustCompile("(?s)```.*?```")
	text = codeBlockRe.ReplaceAllString(text, "")

	inlineCodeRe := regexp.MustCompile("`[^`]+`")
	text = inlineCodeRe.ReplaceAllString(text, "")

	// Remove headers
	headerRe := regexp.MustCompile(`(?m)^#{1,6}\s+(.*)$`)
	text = headerRe.ReplaceAllString(text, "$1")

	// Remove bold/italic
	text = regexp.MustCompile(`\*\*\*(.+?)\*\*\*`).ReplaceAllString(text, "$1")
	text = regexp.MustCompile(`\*\*(.+?)\*\*`).ReplaceAllString(text, "$1")
	text = regexp.MustCompile(`\*(.+?)\*`).ReplaceAllString(text, "$1")
	text = regexp.MustCompile(`__(.+?)__`).ReplaceAllString(text, "$1")
	text = regexp.MustCompile(`_(.+?)_`).ReplaceAllString(text, "$1")

	// Remove bullet points and convert to paragraphs
	bulletRe := regexp.MustCompile(`(?m)^[\s]*[-*+]\s+`)
	text = bulletRe.ReplaceAllString(text, "")

	numberedRe := regexp.MustCompile(`(?m)^[\s]*\d+\.\s+`)
	text = numberedRe.ReplaceAllString(text, "")

	return text
}

// Stage 2: Punctuation Normalization
func (s *Sanitizer) stage2_punctuationNormalization(text string) string {
	if !s.config.NormalizePunct {
		return text
	}

	// Em/en dashes to hyphen or period
	text = strings.ReplaceAll(text, "—", "-")
	text = strings.ReplaceAll(text, "–", "-")

	// Some dashes should be periods
	text = regexp.MustCompile(`\s+-\s+`).ReplaceAllStringFunc(text, func(m string) string {
		if s.rng.Float64() < 0.3 {
			return ". "
		}
		return " - "
	})

	// Smart quotes to ASCII (actual Unicode characters)
	text = strings.ReplaceAll(text, "\u201C", "\"") // left double quote "
	text = strings.ReplaceAll(text, "\u201D", "\"") // right double quote "
	text = strings.ReplaceAll(text, "\u2018", "'")  // left single quote '
	text = strings.ReplaceAll(text, "\u2019", "'")  // right single quote '
	text = strings.ReplaceAll(text, "\u201A", "'")  // single low-9 quote ‚
	text = strings.ReplaceAll(text, "\u201E", "\"") // double low-9 quote „

	// Reduce excessive semicolons
	text = regexp.MustCompile(`;\s+`).ReplaceAllStringFunc(text, func(m string) string {
		if s.rng.Float64() < 0.5 {
			return ". "
		}
		return "; "
	})

	// Reduce colon chaining
	text = regexp.MustCompile(`:\s+`).ReplaceAllStringFunc(text, func(m string) string {
		if s.rng.Float64() < 0.3 {
			return ". "
		}
		return ": "
	})

	return text
}

// Stage 3: Emoji Removal
func (s *Sanitizer) stage3_emojiRemoval(text string) string {
	if !s.config.StripEmojis {
		return text
	}

	var result strings.Builder
	for _, r := range text {
		// Skip emoji ranges (expanded to include flags and newer emojis)
		if r >= 0x1F600 && r <= 0x1F64F || // Emoticons
			r >= 0x1F300 && r <= 0x1F5FF || // Misc Symbols
			r >= 0x1F680 && r <= 0x1F6FF || // Transport
			r >= 0x2600 && r <= 0x26FF || // Misc symbols
			r >= 0x2700 && r <= 0x27BF || // Dingbats
			r >= 0xFE00 && r <= 0xFE0F || // Variation Selectors
			r >= 0x1F900 && r <= 0x1F9FF || // Supplemental Symbols
			r >= 0x1FA70 && r <= 0x1FAFF || // Symbols Extended-A
			r >= 0x1F1E6 && r <= 0x1F1FF { // Regional indicators (flags)
			continue
		}
		result.WriteRune(r)
	}
	return result.String()
}

// Stage 4: LLM Phrase Suppression
func (s *Sanitizer) stage4_llmPhraseSuppression(text string) string {
	if !s.config.RemoveLLMPhrases {
		return text
	}

	// Split into sentences first to avoid eating mid-paragraph content
	sentences := s.splitSentences(text)

	llmPhrases := []string{
		`^In conclusion,?\s*`,
		`^It is important to note that\s+`,
		`^It's important to note that\s+`,
		`^Overall,\s*`,
		`^This highlights the importance of\s+`,
		`^As mentioned earlier,?\s*`,
		`^From a broader perspective,?\s*`,
		`^It's worth noting that\s+`,
		`^It is worth noting that\s+`,
		`^In summary,\s*`,
		`^To summarize,\s*`,
		`^In other words,\s*`,
		`^That being said,\s*`,
		`^Having said that,\s*`,
		`^At the end of the day,\s*`,
		`^The key takeaway is\s+`,
		`^It goes without saying\s+`,
		`^Needless to say,\s*`,
	}

	// CRITICAL: Apply replacements cumulatively, not to original snapshot
	for i := range sentences {
		for _, phrase := range llmPhrases {
			re := regexp.MustCompile(`(?i)` + phrase)
			sentences[i].Text = re.ReplaceAllString(sentences[i].Text, "")
		}
		sentences[i].Text = strings.TrimSpace(sentences[i].Text)
	}

	return s.joinSentences(sentences)
}

// Split text into sentences while preserving punctuation
func (s *Sanitizer) splitSentences(text string) []Sentence {
	// Find all sentence boundaries with punctuation
	re := regexp.MustCompile(`([.!?]+)(\s+|$)`)
	indices := re.FindAllStringIndex(text, -1)

	if len(indices) == 0 {
		return []Sentence{{Text: text, Punctuation: ""}}
	}

	var sentences []Sentence
	lastEnd := 0

	for _, idx := range indices {
		start := idx[0]
		end := idx[1]

		sentText := strings.TrimSpace(text[lastEnd:start])
		punct := strings.TrimSpace(text[start:end])

		if sentText != "" {
			sentences = append(sentences, Sentence{
				Text:        sentText,
				Punctuation: punct,
			})
		}

		lastEnd = end
	}

	// Handle any remaining text
	if lastEnd < len(text) {
		remaining := strings.TrimSpace(text[lastEnd:])
		if remaining != "" {
			sentences = append(sentences, Sentence{
				Text:        remaining,
				Punctuation: "",
			})
		}
	}

	return sentences
}

// Join sentences back together with preserved punctuation
func (s *Sanitizer) joinSentences(sentences []Sentence) string {
	var result strings.Builder
	for i, sent := range sentences {
		if sent.Text == "" {
			continue
		}
		result.WriteString(sent.Text)
		if sent.Punctuation != "" {
			result.WriteString(sent.Punctuation)
		}
		// Add space after punctuation unless it's the last sentence
		if i < len(sentences)-1 && sent.Punctuation != "" {
			result.WriteString(" ")
		}
	}
	return result.String()
}

// Stage 5: Sentence Rhythm De-optimization
func (s *Sanitizer) stage5_sentenceRhythmDeoptimization(text string) string {
	sentences := s.splitSentences(text)
	if len(sentences) < 3 {
		return s.joinSentences(sentences)
	}

	var result []Sentence
	i := 0
	for i < len(sentences) {
		if i < len(sentences)-1 && s.rng.Float64() < 0.15 {
			// Merge two sentences with proper punctuation between them
			merged := Sentence{
				Text:        sentences[i].Text + ". " + sentences[i+1].Text,
				Punctuation: sentences[i+1].Punctuation,
			}
			result = append(result, merged)
			i += 2
		} else if s.rng.Float64() < 0.1 && len(sentences[i].Text) > 100 {
			// Split long sentence (use rune-safe splitting)
			sentRunes := []rune(sentences[i].Text)
			if len(sentRunes) > 100 {
				midpoint := len(sentRunes) / 2
				// Find comma near midpoint
				searchStart := midpoint
				splitIdx := -1
				for j := searchStart; j < len(sentRunes); j++ {
					if sentRunes[j] == ',' {
						splitIdx = j
						break
					}
				}

				if splitIdx > 0 {
					result = append(result, Sentence{
						Text:        strings.TrimSpace(string(sentRunes[:splitIdx])),
						Punctuation: ".",
					})
					result = append(result, Sentence{
						Text:        strings.TrimSpace(string(sentRunes[splitIdx+1:])),
						Punctuation: sentences[i].Punctuation,
					})
				} else {
					result = append(result, sentences[i])
				}
			} else {
				result = append(result, sentences[i])
			}
			i++
		} else {
			result = append(result, sentences[i])
			i++
		}
	}

	return s.joinSentences(result)
}

// Call local model API
func (s *Sanitizer) callLocalModel(prompt string, maxTokens int) (string, error) {
	reqBody := ChatCompletionRequest{
		Model: s.config.ModelName,
		Messages: []ChatMessage{
			{Role: "user", Content: prompt},
		},
		Temperature: s.config.Temperature,
		MaxTokens:   maxTokens,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", s.config.APIEndpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if s.config.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+s.config.APIKey)
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to call API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
	}

	var chatResp ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	return strings.TrimSpace(chatResp.Choices[0].Message.Content), nil
}

// Stage 6: Vocabulary Flattening
func (s *Sanitizer) stage6_vocabularyFlattening(text string) string {
	if s.config.FlattenVocab == "" || s.config.FlattenVocab == "none" {
		return text
	}

	// Static replacements (fallback or supplement)
	staticReplacements := map[string]string{
		"utilize":      "use",
		"Utilize":      "Use",
		"demonstrates": "shows",
		"Demonstrates": "Shows",
		"subsequently": "then",
		"Subsequently": "Then",
		"commence":     "start",
		"Commence":     "Start",
		"terminate":    "end",
		"Terminate":    "End",
		"endeavor":     "try",
		"Endeavor":     "Try",
		"facilitate":   "help with",
		"Facilitate":   "Help with",
		"implement":    "do",
		"Implement":    "Do",
		"optimize":     "improve",
		"Optimize":     "Improve",
		"leverage":     "use",
		"Leverage":     "Use",
	}

	// Apply static replacements
	for formal, casual := range staticReplacements {
		wordBoundary := regexp.MustCompile(`\b` + formal + `\b`)
		text = wordBoundary.ReplaceAllString(text, casual)
	}

	// If local model is enabled, use it for intelligent phrase flattening
	if s.config.UseLocalModel {
		text = s.flattenPhrasesWithModel(text)
	}

	return text
}

// Use local model to flatten formal phrases
func (s *Sanitizer) flattenPhrasesWithModel(text string) string {
	sentences := s.splitSentences(text)
	var flattened []Sentence

	for _, sent := range sentences {
		if sent.Text == "" {
			continue
		}

		// Only process sentences that look formal (have certain markers)
		if s.shouldFlatten(sent.Text) {
			prompt := fmt.Sprintf(`Rewrite this sentence using simpler, more casual language. Keep the same meaning but make it sound less formal and more natural. Only output the rewritten sentence, nothing else:

"%s"`, sent.Text)

			result, err := s.callLocalModel(prompt, 80) // 60-80 tokens for flatten
			if err != nil {
				fmt.Fprintf(os.Stderr, "Warning: model call failed: %v\n", err)
				flattened = append(flattened, sent)
			} else {
				// Clean up the result
				result = strings.Trim(result, `"'`)
				flattened = append(flattened, Sentence{
					Text:        result,
					Punctuation: sent.Punctuation,
				})
			}
		} else {
			flattened = append(flattened, sent)
		}
	}

	return s.joinSentences(flattened)
}

// Determine if sentence should be flattened
func (s *Sanitizer) shouldFlatten(sentence string) bool {
	formalMarkers := []string{
		"furthermore", "moreover", "nevertheless", "consequently",
		"thus", "hence", "whereby", "wherein", "thereof",
		"notwithstanding", "pursuant", "heretofore", "aforementioned",
	}

	lower := strings.ToLower(sentence)
	for _, marker := range formalMarkers {
		if strings.Contains(lower, marker) {
			return true
		}
	}

	// Also flatten if sentence is very long (likely complex)
	return len(sentence) > 150
}

// Stage 7: Partial Rephrasing (with weighted selection)
func (s *Sanitizer) stage7_partialRephrasing(text string) string {
	if s.config.RephraseRatio <= 0 {
		return text
	}

	sentences := s.splitSentences(text)
	if len(sentences) == 0 {
		return text
	}

	// Calculate weights: bias toward early sentences and longer sentences
	weights := make([]float64, len(sentences))
	totalWeight := 0.0
	for i, sent := range sentences {
		// Position weight: earlier sentences get higher weight
		posWeight := 1.0 / (1.0 + float64(i)*0.1)

		// Length weight: longer sentences more likely to be edited
		lengthWeight := 1.0 + math.Log(1.0+float64(len(sent.Text))/100.0)

		weights[i] = posWeight * lengthWeight
		totalWeight += weights[i]
	}

	// Normalize weights
	for i := range weights {
		weights[i] /= totalWeight
	}

	// Select sentences to rephrase using weighted sampling (without replacement)
	numToRephrase := int(float64(len(sentences)) * s.config.RephraseRatio)
	if numToRephrase == 0 && s.config.RephraseRatio > 0 {
		numToRephrase = 1
	}

	toRephrase := make(map[int]bool)
	workingWeights := make([]float64, len(weights))
	copy(workingWeights, weights)

	for i := 0; i < numToRephrase && i < len(sentences); i++ {
		idx := s.weightedChoice(workingWeights)
		toRephrase[idx] = true

		// Zero out the selected weight and renormalize to prevent duplicates
		workingWeights[idx] = 0
		sum := 0.0
		for _, w := range workingWeights {
			sum += w
		}
		if sum > 0 {
			for j := range workingWeights {
				workingWeights[j] /= sum
			}
		}
	}

	var result []Sentence
	for i, sent := range sentences {
		if sent.Text == "" {
			continue
		}

		if toRephrase[i] {
			if s.config.UseLocalModel {
				rephrased := s.rephraseWithModel(sent.Text)
				result = append(result, Sentence{
					Text:        rephrased,
					Punctuation: sent.Punctuation,
				})
			} else {
				result = append(result, Sentence{
					Text:        s.lightRephrase(sent.Text),
					Punctuation: sent.Punctuation,
				})
			}
		} else {
			result = append(result, sent)
		}
	}

	return s.joinSentences(result)
}

// Weighted random selection
func (s *Sanitizer) weightedChoice(weights []float64) int {
	r := s.rng.Float64()
	cumulative := 0.0
	for i, w := range weights {
		cumulative += w
		if r <= cumulative {
			return i
		}
	}
	return len(weights) - 1
}

// Use local model to rephrase
func (s *Sanitizer) rephraseWithModel(sentence string) string {
	prompt := fmt.Sprintf(`Rephrase this sentence slightly while keeping the exact same meaning. Make it sound natural and human-written. Only output the rephrased sentence:

"%s"`, sentence)

	result, err := s.callLocalModel(prompt, 60) // 40-60 tokens for rephrase
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: rephrase failed: %v\n", err)
		return sentence
	}

	return strings.Trim(result, `"'`)
}

func (s *Sanitizer) lightRephrase(sentence string) string {
	// Simple rephrasing: swap some transition words
	transitions := map[string]string{
		"However,":      "But",
		"Therefore,":    "So",
		"Additionally,": "Also",
		"Furthermore,":  "Plus",
		"Moreover,":     "And",
	}

	for old, new := range transitions {
		if strings.HasPrefix(sentence, old) {
			sentence = strings.Replace(sentence, old, new, 1)
			break
		}
	}

	return sentence
}

// Stage 8: Final Pass
func (s *Sanitizer) stage8_finalPass(text string) string {
	// Normalize whitespace
	text = regexp.MustCompile(`[ \t]+`).ReplaceAllString(text, " ")
	text = regexp.MustCompile(`\n{3,}`).ReplaceAllString(text, "\n\n")
	text = strings.TrimSpace(text)

	// Fix spacing around punctuation - only insert space when missing
	text = regexp.MustCompile(`\s+([.,!?;:])`).ReplaceAllString(text, "$1")
	text = regexp.MustCompile(`([.,!?;:])([^\s])`).ReplaceAllString(text, "$1 $2")

	// Fix capitalization after sentence boundaries
	text = regexp.MustCompile(`([.!?]\s+)([a-z])`).ReplaceAllStringFunc(text, func(m string) string {
		runes := []rune(m)
		lastIdx := len(runes) - 1
		runes[lastIdx] = unicode.ToUpper(runes[lastIdx])
		return string(runes)
	})

	// Capitalize first character of entire text
	if len(text) > 0 {
		runes := []rune(text)
		runes[0] = unicode.ToUpper(runes[0])
		text = string(runes)
	}

	return text
}

func main() {
	stripMarkdown := flag.Bool("strip-markdown", true, "Remove markdown formatting")
	stripEmojis := flag.Bool("strip-emojis", true, "Remove emojis")
	normalizePunct := flag.Bool("normalize-punctuation", true, "Normalize punctuation")
	removeLLMPhrases := flag.Bool("remove-llm-phrases", true, "Remove LLM tell phrases")
	flattenVocab := flag.String("flatten-vocab", "medium", "Vocabulary flattening level: none|low|medium|high")
	rephraseRatio := flag.Float64("rephrase-ratio", 0.2, "Ratio of sentences to rephrase (0.0-0.4)")
	domain := flag.String("domain", "casual", "Target domain (reserved for future use)")

	// Local model flags
	useLocalModel := flag.Bool("use-local-model", false, "Use local model for rephrasing and flattening")
	apiEndpoint := flag.String("api-endpoint", "http://localhost:8080/v1/chat/completions", "OpenAI-compatible API endpoint")
	apiKey := flag.String("api-key", "", "API key (optional, for authenticated endpoints)")
	modelName := flag.String("model-name", "model", "Model name to use")
	temperature := flag.Float64("temperature", 0.7, "Model temperature for generation")
	seed := flag.Int64("seed", 0, "Random seed for reproducibility (0 = use current time)")

	flag.Parse()

	config := Config{
		StripMarkdown:    *stripMarkdown,
		StripEmojis:      *stripEmojis,
		NormalizePunct:   *normalizePunct,
		RemoveLLMPhrases: *removeLLMPhrases,
		FlattenVocab:     *flattenVocab,
		RephraseRatio:    *rephraseRatio,
		Domain:           *domain,
		UseLocalModel:    *useLocalModel,
		APIEndpoint:      *apiEndpoint,
		APIKey:           *apiKey,
		ModelName:        *modelName,
		Temperature:      *temperature,
		Seed:             *seed,
	}

	sanitizer := NewSanitizer(config)

	input, err := io.ReadAll(os.Stdin)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
		os.Exit(1)
	}

	output := sanitizer.Sanitize(string(input))
	fmt.Print(output)
}
