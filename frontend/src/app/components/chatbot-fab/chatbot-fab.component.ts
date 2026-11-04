import { Component, signal, inject, computed, ElementRef, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, NavigationEnd } from '@angular/router';
import { LucideAngularModule } from 'lucide-angular';
import { TranslateModule } from '@ngx-translate/core';
import { filter } from 'rxjs/operators';
import { FormsModule } from '@angular/forms';
import { ChatbotService, Message, MessageType, FormType } from '../../services/chatbot.service';

@Component({
  selector: 'app-chatbot-fab',
  standalone: true,
  imports: [CommonModule, LucideAngularModule, FormsModule, TranslateModule],
  templateUrl: './chatbot-fab.component.html',
  styleUrls: ['./chatbot-fab.component.css']
})
export class ChatbotFabComponent {
  private router = inject(Router);
  private chatbotService = inject(ChatbotService);

  isOnChatbotPage = signal<boolean>(false);
  isHovered = signal<boolean>(false);
  isOpen = signal<boolean>(false);
  inputText: string = '';
  toast = signal<string | null>(null);
  showAdminDropdown = signal<boolean>(false);

  adminOptions = [
    'Inscription au semestre',
    "Demande d'attestation",
    'Procédure de rattrapage',
    "Justification d'absence",
    'Demande de stage',
  ];

  /* ── Edit Message State ── */
  editingMessageId = signal<string | null>(null);
  editingText = '';
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  /* ── Delegated signals/computed ── */
  isTyping = this.chatbotService.isTyping;

  messages = computed(() => {
    const list = this.chatbotService.messages();
    if (list.length === 0) {
      // simulated welcome message when discussion is brand new
      return [{
        id: 'welcome-msg',
        text: 'Bonjour Jone ! Je suis votre assistant UniHelp. Comment puis-je vous aider ?',
        sender: 'ai' as const,
        timestamp: new Date(),
        suggestions: ['Résumer un document', 'Question administrative']
      } as Message];
    }
    return list;
  });

  constructor() {
    // Check initial route
    this.isOnChatbotPage.set(this.router.url.includes('/student/chatbot'));

    // Listen for route changes
    this.router.events.pipe(
      filter((event): event is NavigationEnd => event instanceof NavigationEnd)
    ).subscribe((event) => {
      this.isOnChatbotPage.set(event.urlAfterRedirects.includes('/student/chatbot'));
    });
  }

  goToChatbot() {
    this.isOpen.set(false);
    this.router.navigate(['/student/chatbot']);
  }

  toggleChat() {
    console.log('toggleChat clicked. Current isOpen state:', this.isOpen());
    this.isOpen.update(v => !v);
    console.log('New isOpen state:', this.isOpen());
    if (this.isOpen()) {
      this.isHovered.set(false);
      this.scrollToBottom();
    } else {
      this.showAdminDropdown.set(false);
    }
  }

  onMouseEnter() {
    this.isHovered.set(true);
  }

  onMouseLeave() {
    this.isHovered.set(false);
  }

  sendMessage(text?: string) {
    const messageText = text || this.inputText.trim();
    if (!messageText) return;

    this.showAdminDropdown.set(false);
    this.chatbotService.sendMessage(messageText);

    if (!text) {
      this.inputText = '';
    }

    this.scrollToBottom();
  }

  /**
   * Handles suggestion chip clicks. If the suggestion is to summarize a document,
   * opens a file selector, otherwise forwards as a normal message.
   */
  handleSuggestion(suggestion: string) {
    const lowered = suggestion.toLowerCase();
    if (lowered.includes('résumer') && lowered.includes('document')) {
      // Trigger file selection for summarization
      this.fileInput.nativeElement.value = '';
      this.fileInput.nativeElement.click();
    } else {
      this.sendMessage(suggestion);
    }
  }

  toggleAdminDropdown(e: Event) {
    e.stopPropagation();
    this.showAdminDropdown.update(v => !v);
  }

  closeAdminDropdown(e: Event) {
    e.stopPropagation();
    this.showAdminDropdown.set(false);
  }

  selectAdminOption(option: string, e: Event) {
    e.stopPropagation();
    this.showAdminDropdown.set(false);
    this.sendMessage(option);
  }

  /**
   * Reads the selected file and sends its content to the chatbot for summarization.
   */
  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      const file = input.files[0];
      // Send only the file name as the message content
      const msg = `Document sélectionné : ${file.name}`;
      this.sendMessage(msg);
    }
  }

  copyMessage(text: string) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(() => {
        this.showToast('Copié !');
      }).catch(err => {
        this.fallbackCopyText(text);
      });
    } else {
      this.fallbackCopyText(text);
    }
  }

  fallbackCopyText(text: string) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.top = "0";
    textArea.style.left = "0";
    textArea.style.position = "fixed";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    try {
      document.execCommand('copy');
      this.showToast('Copié !');
    } catch (err) {
      console.error('Fallback copy failed', err);
      this.showToast('Erreur lors de la copie');
    }
    document.body.removeChild(textArea);
  }

  readAloud() {
    this.showToast('Lecture audio...');
  }

  refreshMessage(msgId: string) {
    this.chatbotService.regenerateMessage(msgId);
    this.showToast("Régénération du message...");
  }

  /* ── Form actions ── */
  updateFormField(msgId: string, fieldName: string, value: string) {
    this.chatbotService.updateFormField(msgId, fieldName, value);
  }

  submitForm(msg: Message) {
    this.chatbotService.submitForm(msg);
    this.scrollToBottom();
  }

  getFormTitle(formType: FormType | undefined): string {
    if (formType === 'attestation') return "Demande d'attestation";
    if (formType === 'reclamation') return 'Réclamation de note';
    if (formType === 'stage')       return 'Demande de stage';
    if (formType === 'absence')     return "Justification d'absence";
    return 'Formulaire';
  }

  /* ── Edit Message actions ── */
  editMessage(msg: Message) {
    this.editingMessageId.set(msg.id);
    this.editingText = msg.text;
  }

  cancelEdit() {
    this.editingMessageId.set(null);
    this.editingText = '';
  }

  submitEdit(msgId: string) {
    if (!this.editingText.trim()) return;
    this.chatbotService.submitEdit(msgId, this.editingText);
    this.cancelEdit();
    this.scrollToBottom();
  }

  showToast(msg: string) {
    this.toast.set(msg);
    setTimeout(() => this.toast.set(null), 2000);
  }

  getTypeBadgeClass(type: MessageType | undefined): string {
    if (type === 'resume') return 'badge-resume';
    if (type === 'question') return 'badge-question';
    if (type === 'form') return 'badge-form';
    return 'badge-normal';
  }

  getTypeBadgeLabel(type: MessageType | undefined): string {
    if (type === 'resume') return 'RÉSUMÉ DÉTECTÉ';
    if (type === 'question') return 'QUESTION DÉTECTÉE';
    if (type === 'form') return 'FORMULAIRE DÉTECTÉ';
    return '';
  }

  formatTime(date: Date | any): string {
    const d = date instanceof Date ? date : new Date(date);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  private scrollToBottom() {
    // Single scroll after the next tick to allow the DOM to render new messages.
    setTimeout(() => {
      const chatContainer = document.querySelector('.chat-messages');
      if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }
    }, 0);
  }
}
