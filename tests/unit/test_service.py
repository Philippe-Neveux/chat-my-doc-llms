from unittest.mock import Mock, patch

from chat_my_doc_llms.service import EXAMPLE_INPUT, Summarization


class TestSummarization:
    @patch('chat_my_doc_llms.service.pipeline')
    def test_summarization_init(self, mock_pipeline):
        mock_pipeline.return_value = Mock()
        
        service = Summarization()
        
        mock_pipeline.assert_called_once_with('summarization')
        assert service.pipeline is not None

    @patch('chat_my_doc_llms.service.pipeline')
    def test_summarize_with_default_input(self, mock_pipeline):
        mock_summarizer = Mock()
        mock_summarizer.return_value = [{'summary_text': 'This is a test summary'}]
        mock_pipeline.return_value = mock_summarizer
        
        service = Summarization()
        result = service.summarize()
        
        assert result == "Hello world! Here's your summary: This is a test summary"
        mock_summarizer.assert_called_once_with(EXAMPLE_INPUT)

    @patch('chat_my_doc_llms.service.pipeline')
    def test_summarize_with_custom_input(self, mock_pipeline):
        custom_text = "This is a custom text to summarize"
        mock_summarizer = Mock()
        mock_summarizer.return_value = [{'summary_text': 'Custom summary'}]
        mock_pipeline.return_value = mock_summarizer
        
        service = Summarization()
        result = service.summarize(custom_text)
        
        assert result == "Hello world! Here's your summary: Custom summary"
        mock_summarizer.assert_called_once_with(custom_text)

    @patch('chat_my_doc_llms.service.pipeline')
    def test_summarize_empty_input(self, mock_pipeline):
        mock_summarizer = Mock()
        mock_summarizer.return_value = [{'summary_text': 'Empty summary'}]
        mock_pipeline.return_value = mock_summarizer
        
        service = Summarization()
        result = service.summarize("")
        
        assert result == "Hello world! Here's your summary: Empty summary"
        mock_summarizer.assert_called_once_with("")

    def test_example_input_constant(self):
        assert isinstance(EXAMPLE_INPUT, str)
        assert len(EXAMPLE_INPUT) > 0
        assert "Breaking News" in EXAMPLE_INPUT
        assert "Whiskers" in EXAMPLE_INPUT