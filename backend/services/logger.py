import os
from datetime import datetime
from typing import Optional, List, Dict, Any

class TranslationLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        self.current_log_file: Optional[str] = None
        self._is_active = False

    def start_file_session(self, filename: str):
        """Starts a logging session for a specific file."""
        # Create a log file named after the PDF or with a timestamp
        safe_filename = os.path.basename(filename).replace("/", "_").replace("\\", "_")
        self.current_log_file = os.path.join(self.log_dir, f"{safe_filename}.logs")
        
        timestamp = datetime.now().strftime("%d.%m.%y - %H:%M:%S")
        with open(self.current_log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"FILENAME: {os.path.basename(filename)}\n")
            f.write(f"TRANSLATION STARTED: {timestamp}\n")
            f.write("="*80 + "\n\n")
        self._is_active = True

    def log_entry(self, 
                  source_text: str, 
                  paragraph_group: Optional[str] = None,
                  inline_blocks: Optional[List[str]] = None,
                  manual_translation: Optional[str] = None,
                  llm1_translation: Optional[str] = None,
                  llm2_translation: Optional[str] = None,
                  glossary_term: Optional[str] = None,
                  final_text: Optional[str] = None,
                  skipped: bool = False,
                  skip_reason: Optional[str] = None,
                  insights: Optional[Dict[str, Any]] = None):
        """Logs a single translation entry with various refined stages."""
        
        if not self._is_active or not self.current_log_file:
            return

        timestamp = datetime.now().strftime("%d.%m.%y - %H:%M:%S")
        
        with open(self.current_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}]\n")
            f.write(f"SOURCE TEXT: {source_text}\n")
            
            if skipped:
                f.write(f"STATUS: SKIPPED\n")
                if skip_reason:
                    f.write(f"REASON: {skip_reason}\n")
            else:
                if paragraph_group:
                    f.write(f"PARAGRAPH GROUP (DELIMITED): {paragraph_group}\n")
                if inline_blocks:
                    # If any inline blocks were processed
                    f.write(f"INLINE BLOCKS: {', '.join(inline_blocks)}\n")
                
                if manual_translation:
                    f.write(f"MT (AZURE): {manual_translation}\n")
                
                if llm1_translation:
                    f.write(f"LLM-1 REFINED: {llm1_translation}\n")
                
                llm2_content = f"LLM-2 REFINED: {llm2_translation}" if llm2_translation else ""
                if glossary_term:
                    llm2_content += f" (TOP GLOSSARY: {glossary_term})"
                
                if llm2_content:
                    f.write(llm2_content + "\n")
                
                if final_text:
                    f.write(f"FINAL RESTORED: {final_text}\n")
            
            if insights:
                f.write("INSIGHTS:\n")
                for k, v in insights.items():
                    f.write(f"  - {k}: {v}\n")
            
            f.write("-" * 40 + "\n\n")

    def log_general_insights(self, insights: str):
        """Logs general information about the process."""
        if not self._is_active or not self.current_log_file:
            return
            
        with open(self.current_log_file, "a", encoding="utf-8") as f:
            f.write(f"[GENERAL INSIGHTS]\n{insights}\n\n")

# Global singleton or instance management
_logger_instance = TranslationLogger()

def get_logger() -> TranslationLogger:
    return _logger_instance
