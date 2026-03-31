import os
from datetime import datetime
from typing import Optional, List, Dict, Any

class TranslationLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        self.current_log_file: Optional[str] = None
        self.base_filename: Optional[str] = None
        self.file_index = 1
        self.line_count = 0
        self._is_active = False

    def _rotate_log_if_needed(self):
        if self.line_count >= 250000 and self.base_filename:
            self.file_index += 1
            self.current_log_file = os.path.join(self.log_dir, f"{self.base_filename}_{self.file_index}.logs")
            self.line_count = 0
            timestamp = datetime.now().strftime("%d.%m.%y - %H:%M:%S")
            lines = [
                "\n" + "="*80,
                f"FILENAME: {self.base_filename} (Part {self.file_index})",
                f"TRANSLATION CONTINUED: {timestamp}",
                "="*80 + "\n\n"
            ]
            with open(self.current_log_file, "a", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.line_count += len(lines)

    def start_file_session(self, filename: str):
        """Starts a logging session for a specific file."""
        self.base_filename = os.path.basename(filename).replace("/", "_").replace("\\", "_")
        self.file_index = 1
        self.current_log_file = os.path.join(self.log_dir, f"{self.base_filename}.logs")
        self.line_count = 0
        
        timestamp = datetime.now().strftime("%d.%m.%y - %H:%M:%S")
        lines = [
            "\n" + "="*80,
            f"FILENAME: {os.path.basename(filename)}",
            f"TRANSLATION STARTED: {timestamp}",
            "="*80 + "\n\n"
        ]
        with open(self.current_log_file, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self.line_count += len(lines)
        self._is_active = True

    def log_entry(self, 
                  source_text: str, 
                  chunk_info: Optional[tuple] = None,
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

        self._rotate_log_if_needed()

        timestamp = datetime.now().strftime("%d.%m.%y - %H:%M:%S")
        
        lines = []
        if chunk_info:
            lines.append(f"[{timestamp}] ({chunk_info[0]}/{chunk_info[1]})")
        else:
            lines.append(f"[{timestamp}]")
            
        lines.append(f"SOURCE TEXT: {source_text}")
        
        if skipped:
            lines.append(f"STATUS: SKIPPED")
            if skip_reason:
                lines.append(f"REASON: {skip_reason}")
        else:
            if paragraph_group:
                lines.append(f"PARAGRAPH GROUP (DELIMITED): {paragraph_group}")
            if inline_blocks:
                lines.append(f"INLINE BLOCKS: {', '.join(inline_blocks)}")
            
            if manual_translation:
                lines.append(f"MT (AZURE): {manual_translation}")
            
            if llm1_translation:
                lines.append(f"LLM-1 REFINED: {llm1_translation}")
            
            llm2_content = f"LLM-2 REFINED: {llm2_translation}" if llm2_translation else ""
            if glossary_term:
                llm2_content += f" (TOP GLOSSARY: {glossary_term})"
            
            if llm2_content:
                lines.append(llm2_content)
            
            if final_text is not None:
                lines.append(f"FINAL RESTORED: {final_text}")
        
        if insights:
            lines.append("INSIGHTS:")
            for k, v in insights.items():
                if isinstance(v, list):
                    lines.append(f"  - {k}:")
                    for item in v:
                        lines.append(f"      * {item}")
                else:
                    lines.append(f"  - {k}: {v}")
        
        lines.append("-" * 40 + "\n")
        
        with open(self.current_log_file, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            
        self.line_count += len(lines) + 1

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
