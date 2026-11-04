import {
  Component,
  signal,
  computed,
  ElementRef,
  ViewChild,
  AfterViewChecked,
  AfterViewInit,
  inject,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { TranslateModule } from '@ngx-translate/core';
import { animate, style, transition, trigger } from '@angular/animations';
import { ChatbotService, Message, Conversation, MessageType, FormType, FormField } from '../../../services/chatbot.service';

/* ─── Component ──────────────────────────────────────────────────────────── */
@Component({
  selector: 'app-chatbot',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule, TranslateModule],
  templateUrl: './chatbot.component.html',
  styleUrls: ['./chatbot.component.css'],
  animations: [
    trigger('fadeSlideDown', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translate(-50%, -20px)' }),
        animate('250ms ease-out', style({ opacity: 1, transform: 'translate(-50%, 0)' })),
      ]),
      transition(':leave', [
        animate('200ms ease-in', style({ opacity: 0, transform: 'translate(-50%, -20px)' })),
      ]),
    ]),
    trigger('fadeSlideUp', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(10px)' }),
        animate('200ms ease-out', style({ opacity: 1, transform: 'translateY(0)' })),
      ]),
      transition(':leave', [
        animate('150ms ease-in', style({ opacity: 0, transform: 'translateY(10px)' })),
      ]),
    ]),
    trigger('msgIn', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(40px)' }),
        animate('500ms cubic-bezier(0.34, 1.56, 0.64, 1)', style({ opacity: 1, transform: 'translateY(0)' })),
      ]),
    ]),
    trigger('chipIn', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(4px)' }),
        animate('200ms 200ms ease-out', style({ opacity: 1, transform: 'translateY(0)' })),
      ]),
    ]),
  ],
})
export class ChatbotComponent implements AfterViewChecked, AfterViewInit {
  @ViewChild('scrollContainer') scrollContainer!: ElementRef;
  @ViewChild('textarea') textareaRef!: ElementRef;

  private chatbotService = inject(ChatbotService);

  /* ── Local Component UI State ── */
  inputValue = '';
  toast = signal<string | null>(null);
  showAdminDropdown = signal<boolean>(false);
  contextMenuId = signal<string | null>(null);
  renamingId = signal<string | null>(null);
  renamingValue = '';
  /* ── Edit Message State ── */
  editingMessageId = signal<string | null>(null);
  editingText = '';
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  /* ── Delegated signals from ChatbotService ── */
  conversations = this.chatbotService.conversations;
  activeConvId = this.chatbotService.activeConvId;
  isTyping = this.chatbotService.isTyping;
  searchQuery = this.chatbotService.searchQuery;

  activeConversation = this.chatbotService.activeConversation;
  messages = this.chatbotService.messages;
  filteredConversations = this.chatbotService.filteredConversations;
  favoriteConversations = this.chatbotService.favoriteConversations;
  todayConversations = this.chatbotService.todayConversations;
  hierConversations = this.chatbotService.hierConversations;
  anciennesConversations = this.chatbotService.anciennesConversations;
  showEmptyState = this.chatbotService.showEmptyState;

  /* ── Admin options ── */
  adminOptions = [
    'Inscription au semestre',
    "Demande d'attestation",
    'Procédure de rattrapage',
    "Justification d'absence",
    'Demande de stage',
  ];

  /* ── Lifecycle ── */
  private shouldScroll = false;

  ngAfterViewInit() {
  }

  ngAfterViewChecked() {
    if (this.shouldScroll) {
      this.scrollToBottom();
      this.shouldScroll = false;
    }
  }

  /* ── Scroll ── */
  scrollToBottom() {
    if (this.scrollContainer) {
      const el = this.scrollContainer.nativeElement;
      el.scrollTop = el.scrollHeight;
    }
  }

  /* ── Textarea ── */
  adjustHeight() {
    if (this.textareaRef) {
      const el = this.textareaRef.nativeElement;
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 150) + 'px';
    }
  }

  handleEnter(e: Event) {
    const ev = e as KeyboardEvent;
    if (ev.key === 'Enter' && !ev.shiftKey) {
      ev.preventDefault();
      this.handleSend();
    }
  }

  handleSend() {
    if (!this.inputValue.trim() || this.isTyping()) return;
    this.sendMessage(this.inputValue.trim());
    this.inputValue = '';
    setTimeout(() => this.adjustHeight(), 0);
  }

  /* ── Send message ── */
  sendMessage(text: string) {
    if (!text.trim()) return;
    this.showAdminDropdown.set(false);
    this.chatbotService.sendMessage(text);
    this.shouldScroll = true;
  }

  /**
   * Handles suggestion chip clicks. If the suggestion is to summarize a document,
   * opens the hidden file selector, otherwise forwards as a normal message.
   */
  handleSuggestion(suggestion: string) {
    const lowered = suggestion.toLowerCase();
    if (lowered.includes('résumer') && lowered.includes('document')) {
      this.fileInput.nativeElement.value = '';
      this.fileInput.nativeElement.click();
    } else {
      this.sendMessage(suggestion);
    }
  }

  /**
   * Reads the selected file and sends only its name as a chat message.
   */
  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      const file = input.files[0];
      const msg = `Document sélectionné : ${file.name}`;
      this.sendMessage(msg);
    }
  }

  /* ── Form actions ── */
  updateFormField(msgId: string, fieldName: string, value: string) {
    this.chatbotService.updateFormField(msgId, fieldName, value);
  }

  getFormData(msgId: string): Record<string, string> {
    return this.chatbotService.getFormData(msgId);
  }

  submitForm(msg: Message) {
    this.chatbotService.submitForm(msg);
    this.shouldScroll = true;
  }

  /* ── Conversation management ── */
  startNewConversation() {
    this.chatbotService.startNewConversation();
    this.inputValue = '';
    this.showAdminDropdown.set(false);
    this.cancelEdit();
  }

  setActiveConv(id: string) {
    this.chatbotService.setActiveConv(id);
    this.shouldScroll = true;
    this.cancelEdit();
  }

  isDeleteModalOpen = signal(false);
  deletingConvId = signal<string | null>(null);

  deleteConversation(id: string) {
    this.deletingConvId.set(id);
    this.isDeleteModalOpen.set(true);
    this.contextMenuId.set(null);
  }

  confirmDelete() {
    const id = this.deletingConvId();
    if (id) {
      this.chatbotService.deleteConversation(id);
      this.showToast('Conversation supprimée');
      this.cancelEdit();
    }
    this.isDeleteModalOpen.set(false);
    this.deletingConvId.set(null);
  }

  cancelDelete() {
    this.isDeleteModalOpen.set(false);
    this.deletingConvId.set(null);
  }

  toggleFavorite(id: string) {
    this.chatbotService.toggleFavorite(id);
    this.contextMenuId.set(null);
    this.showToast('Favoris mis à jour');
  }

  /* ── Rename ── */
  startRenaming(convId: string, currentTitle: string) {
    this.contextMenuId.set(null);
    this.renamingId.set(convId);
    this.renamingValue = currentTitle;
  }

  finishRenaming() {
    const id = this.renamingId();
    if (id && this.renamingValue.trim()) {
      this.chatbotService.renameConversation(id, this.renamingValue.trim());
      this.showToast('Conversation renommée');
    }
    this.renamingId.set(null);
    this.renamingValue = '';
  }

  handleRenameKey(e: KeyboardEvent) {
    if (e.key === 'Enter') { e.preventDefault(); this.finishRenaming(); }
    if (e.key === 'Escape') { this.renamingId.set(null); this.renamingValue = ''; }
  }

  /* ── Context menu ── */
  toggleContextMenu(id: string, e: Event) {
    e.stopPropagation();
    this.contextMenuId.set(this.contextMenuId() === id ? null : id);
  }

  closeContextMenu() { this.contextMenuId.set(null); }

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

  readAloud() { this.showToast('Lecture audio...'); }
  refreshMessage(msgId: string) {
    this.chatbotService.regenerateMessage(msgId);
    this.showToast("Régénération du message...");
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
    this.shouldScroll = true;
  }

  showToast(msg: string) {
    this.toast.set(msg);
    setTimeout(() => this.toast.set(null), 2500);
  }

  formatTime(date: Date | any): string {
    const d = date instanceof Date ? date : new Date(date);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  getTypeBadgeClass(type: MessageType | undefined): string {
    if (type === 'question') return 'bg-emerald-100 text-emerald-700';
    if (type === 'form')     return 'bg-blue-100 text-blue-700';
    return 'bg-purple-100 text-purple-700';
  }

  getTypeBadgeLabel(type: MessageType | undefined): string {
    if (type === 'question') return 'Question détectée';
    if (type === 'form')     return 'Formulaire détecté';
    return 'Résumé détecté';
  }

  getFormTitle(formType: FormType | undefined): string {
    if (formType === 'attestation') return "Demande d'attestation";
    if (formType === 'reclamation') return 'Réclamation de note';
    if (formType === 'stage')       return 'Demande de stage';
    if (formType === 'absence')     return "Justification d'absence";
    return 'Formulaire';
  }

  /* Track-by helpers */
  trackById(_: number, item: { id: string }) { return item.id; }
  trackByIdx(i: number) { return i; }
}
