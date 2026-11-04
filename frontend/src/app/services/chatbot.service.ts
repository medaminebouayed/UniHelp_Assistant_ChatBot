import { Injectable, signal, computed, inject, effect } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import { environment } from '../../environments/environment';
import { AuthService } from './auth.service';
import { ConversationService, BackendConversation } from './conversation.service';

/* ─── Types ─────────────────────────────────────────────────────────────── */
export type MessageType = 'normal' | 'question' | 'resume' | 'form';
export type FormType = 'attestation' | 'reclamation' | 'stage' | 'absence';

export interface FormField {
  name: string;
  label: string;
  type: 'text' | 'email' | 'number' | 'date' | 'select' | 'textarea';
  placeholder?: string;
  options?: string[];
  required?: boolean;
}

export interface SourceInfo {
  id: string;
  site?: string;
  title?: string;
  section?: string;
  type?: string;
  url?: string;
}

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  type?: MessageType;
  source?: string;
  sources?: SourceInfo[];
  suggestions?: string[];
  formType?: FormType;
  formFields?: FormField[];
  formSubmitted?: boolean;
  formData?: Record<string, string>;
}

export interface Conversation {
  id: string;
  title: string;
  date: string;
  isFavorite?: boolean;
  messages: Message[];
}

/* ─── Pure helpers (unchanged from original) ────────────────────────────── */
export function getSuggestions(type: MessageType): string[] {
  if (type === 'question' || type === 'form')
    return ['Quels documents fournir ?', 'Quelles sont les dates limites ?', 'Qui contacter ?'];
  return ['Aide avec mes devoirs', 'Question administrative', 'Résumer un document'];
}

export function getSource(type: MessageType): string {
  if (type === 'question' || type === 'form') return 'Source : Règlement intérieur, Art. 15';
  if (type === 'resume') return "Source : Guide de l'étudiant, p. 42";
  return 'Source : Base de connaissances UniHelp';
}

export function getFormFields(type: FormType): FormField[] {
  switch (type) {
    case 'attestation':
      return [
        { name: 'nom',        label: 'Nom complet',           type: 'text',     required: true, placeholder: 'Ex: Jean Dupont' },
        { name: 'filiere',    label: 'Filière',               type: 'text',     required: true, placeholder: 'Ex: Informatique L3' },
        { name: 'annee',      label: 'Année universitaire',   type: 'select',   required: true, options: ['2023-2024','2024-2025','2025-2026'] },
        { name: 'numEtudiant',label: 'N° étudiant',           type: 'text',     required: true, placeholder: 'Ex: 20240156' },
        { name: 'motif',      label: 'Motif',                 type: 'textarea', required: true, placeholder: 'Pourquoi avez-vous besoin de cette attestation ?' },
      ];
    case 'reclamation':
      return [
        { name: 'nom',         label: 'Nom complet',          type: 'text',     required: true },
        { name: 'matiere',     label: 'Matière',              type: 'text',     required: true },
        { name: 'typeEval',    label: "Type d'évaluation",    type: 'select',   required: true, options: ['Examen','Contrôle continu','TP','TD'] },
        { name: 'dateEpreuve', label: "Date de l'épreuve",    type: 'date',     required: true },
        { name: 'noteObtenue', label: 'Note obtenue',         type: 'number',   required: true },
        { name: 'noteAttendue',label: 'Note attendue',        type: 'number',   required: true },
        { name: 'motif',       label: 'Motif de réclamation', type: 'textarea', required: true, placeholder: 'Expliquez précisément les raisons...' },
      ];
    case 'stage':
      return [
        { name: 'nom',        label: 'Nom complet',           type: 'text',     required: true },
        { name: 'filiere',    label: 'Filière',               type: 'text',     required: true },
        { name: 'entreprise', label: 'Entreprise / Organisme',type: 'text',     required: true },
        { name: 'periode',    label: 'Période souhaitée',     type: 'text',     required: true, placeholder: 'Ex: du 01/06 au 31/08' },
        { name: 'objet',      label: 'Objet / Description',   type: 'textarea', required: true, placeholder: 'Décrivez brièvement le stage...' },
      ];
    case 'absence':
      return [
        { name: 'nom',     label: 'Nom complet',             type: 'text',     required: true },
        { name: 'dates',   label: "Date(s) d'absence",       type: 'text',     required: true, placeholder: 'Ex: 12 et 13 Octobre' },
        { name: 'matieres',label: 'Matière(s) concernée(s)', type: 'text',     required: true },
        { name: 'motif',   label: 'Motif',                   type: 'select',   required: true, options: ['Maladie','Raison familiale','Convocation officielle','Autre'] },
        { name: 'details', label: 'Détails supplémentaires', type: 'textarea', required: true },
      ];
    default:
      return [];
  }
}

export function generateDocument(formType: FormType, data: Record<string, string>): string {
  if (formType === 'attestation') {
    return `Voici votre demande d'attestation de scolarité formatée :\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\nObjet : Demande d'attestation de scolarité\n\nMadame, Monsieur,\n\nJe soussigné(e) ${data['nom'] || '[Nom]'}, étudiant(e) en ${data['annee'] || '[Année]'} ${data['filiere'] || '[Filière]'} (N° étudiant : ${data['numEtudiant'] || '[N°]'}), me permets de solliciter auprès de vos services la délivrance d'une attestation de scolarité pour l'année universitaire en cours.\n\nCette attestation m'est nécessaire pour : ${data['motif'] || '[motif]'}.\n\nJe vous remercie de bien vouloir traiter cette demande dans les meilleurs délais.\n\nVeuillez agréer, Madame, Monsieur, l'expression de mes salutations distinguées.\n\n${data['nom'] || '[Nom]'}\nN° Étudiant: ${data['numEtudiant'] || '[N°]'}\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\nVous pouvez copier ce modèle et l'envoyer au secrétariat.`;
  } else if (formType === 'reclamation') {
    return `Voici votre formulaire de réclamation de note :\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\nFORMULAIRE DE RÉCLAMATION DE NOTE\n\nObjet : Réclamation concernant une note d'examen\n\nMadame, Monsieur,\n\nJe soussigné(e) ${data['nom'] || '[Nom]'}, souhaite formuler une réclamation concernant :\n\n• Matière : ${data['matiere'] || '[Matière]'}\n• Type d'évaluation : ${data['typeEval'] || '[Type]'}\n• Date de l'épreuve : ${data['dateEpreuve'] || '[Date]'}\n• Note obtenue : ${data['noteObtenue'] || '[Note]'}\n• Note attendue : ${data['noteAttendue'] || '[Note]'}\n\nMotif de la réclamation :\n${data['motif'] || '[Motif]'}\n\nCordialement,\n\n${data['nom'] || '[Nom]'}\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\nVous pouvez copier ce modèle et l'envoyer à votre professeur ou au secrétariat.`;
  } else if (formType === 'stage') {
    return `Voici votre demande de stage formatée :\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\nObjet : Demande de convention de stage - ${data['entreprise'] || '[Entreprise]'}\n\nMadame, Monsieur,\n\nJe soussigné(e) ${data['nom'] || '[Nom]'}, étudiant(e) en ${data['filiere'] || '[Filière]'}, vous informe avoir trouvé un stage au sein de l'entreprise ${data['entreprise'] || '[Entreprise]'}.\n\nCe stage se déroulera sur la période suivante : ${data['periode'] || '[Période]'}.\n\nDescription du stage :\n${data['objet'] || '[Description]'}\n\nCordialement,\n\n${data['nom'] || '[Nom]'}\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━`;
  } else if (formType === 'absence') {
    return `Voici votre justification d'absence formatée :\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\nObjet : Justification d'absence - ${data['nom'] || '[Nom]'}\n\nMadame, Monsieur,\n\nJe vous prie de bien vouloir excuser mon absence du ${data['dates'] || '[Dates]'} concernant le(s) cours de ${data['matieres'] || '[Matières]'}.\n\nCette absence est due à la raison suivante : ${data['motif'] || '[Motif]'}.\n\nDétails : ${data['details'] || '[Détails]'}\n\nCordialement,\n\n${data['nom'] || '[Nom]'}\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━`;
  }
  return '';
}

@Injectable({ providedIn: 'root' })
export class ChatbotService {
  private readonly API = environment.apiUrl;
  private http = inject(HttpClient);
  private authService = inject(AuthService);
  private convService = inject(ConversationService);

  conversations = signal<Conversation[]>([]);
  activeConvId = signal<string | null>(null);
  isTyping = signal<boolean>(false);
  searchQuery = signal<string>('');

  private _lastLoadedUserId: number | null = null;

  constructor() {
    // Use effect() so that loadConversations() is called reactively
    // AFTER AuthService.user() signal resolves (async fetchMe completes).
    // This fixes the race condition where user() was null at construction time.
    effect(() => {
      const user = this.authService.user();
      if (user && user.id !== this._lastLoadedUserId) {
        this._lastLoadedUserId = user.id;
        this.loadConversations();
      }
      if (!user) {
        this._lastLoadedUserId = null;
        this.conversations.set([]);
      }
    }, { allowSignalWrites: true });
  }

  /* ── Computed state ── */
  activeConversation = computed(() =>
    this.conversations().find(c => c.id === this.activeConvId())
  );

  messages = computed(() => this.activeConversation()?.messages || []);

  filteredConversations = computed(() => {
    const q = this.searchQuery().toLowerCase();
    return this.conversations().filter(c => c.title.toLowerCase().includes(q));
  });

  favoriteConversations = computed(() =>
    this.filteredConversations().filter(c => c.isFavorite)
  );

  todayConversations = computed(() =>
    this.filteredConversations().filter(c => c.date === "Aujourd'hui" && !c.isFavorite)
  );

  hierConversations = computed(() =>
    this.filteredConversations().filter(c => c.date === 'Hier' && !c.isFavorite)
  );

  anciennesConversations = computed(() =>
    this.filteredConversations().filter(c => c.date !== "Aujourd'hui" && c.date !== 'Hier' && !c.isFavorite)
  );

  showEmptyState = computed(() =>
    !this.activeConvId() && this.messages().length === 0
  );

  /* ── Load conversations from backend ── */
  async loadConversations(): Promise<void> {
    const user = this.authService.user();
    if (!user) return;
    try {
      const backendConvs = await firstValueFrom(
        this.convService.getByUser(user.id)
      );
      this.conversations.set(
        backendConvs.map((bc) => this.convService.mapBackendToFrontend(bc)) as any
      );
    } catch {
      this.conversations.set([]);
    }
  }

  /* ── Actions ── */
  async startNewConversation(): Promise<void> {
    this.activeConvId.set(null);
  }

  setActiveConv(id: string): void {
    this.activeConvId.set(id);
  }

  async deleteConversation(id: string): Promise<void> {
    await firstValueFrom(this.convService.delete(Number(id)));
    this.conversations.update(list => list.filter(c => c.id !== id));
    if (this.activeConvId() === id) this.activeConvId.set(null);
  }

  async toggleFavorite(id: string): Promise<void> {
    await firstValueFrom(this.convService.toggleFavorite(Number(id)));
    this.conversations.update(list =>
      list.map(c => c.id === id ? { ...c, isFavorite: !c.isFavorite } : c)
    );
  }

  async renameConversation(id: string, newTitle: string): Promise<void> {
    this.conversations.update(list =>
      list.map(c => c.id === id ? { ...c, title: newTitle } : c)
    );
  }

  detectFormType(text: string): FormType | null {
    const l = text.toLowerCase();
    // 'scolarite' alone is too broad (catches "frais de scolarite" questions)
    // Require explicit request keywords near it
    if (l.includes('attestation')) return 'attestation';
    if ((l.includes('scolarité') || l.includes('scolaire')) &&
        (l.includes('demande') || l.includes('certificat') || l.includes('attestation') ||
         l.includes('obtenir') || l.includes('avoir') || l.includes('formulaire'))) return 'attestation';
    if (l.includes('réclam') || l.includes('reclam') || (l.includes('note') && !l.includes('frais'))) return 'reclamation';
    if (l.includes('stage') || l.includes('convention')) return 'stage';
    if (l.includes('absence') || l.includes('justif')) return 'absence';
    return null;
  }

  detectType(text: string): MessageType {
    const l = text.toLowerCase();
    if (l.includes('résume') || l.includes('synthèse') || l.includes('résumé')) return 'resume';
    if (l.includes('comment') || l.includes('pourquoi') || l.includes('quel') || text.includes('?')) return 'question';
    return 'normal';
  }

  async sendMessage(text: string, skipAddingUserMsg = false): Promise<void> {
    if (!text.trim()) return;
    const user = this.authService.user();
    if (!user) return;

    let currentConvId = this.activeConvId();

    if (!skipAddingUserMsg) {
      const userMsg: Message = {
        id: 'msg-' + Date.now(),
        text: text.trim(),
        sender: 'user',
        timestamp: new Date(),
      };

      if (!currentConvId) {
        const newConvId = 'conv-' + Date.now();
        const newConv: Conversation = {
          id: newConvId,
          title: text.trim().substring(0, 30) + (text.trim().length > 30 ? '...' : ''),
          date: "Aujourd'hui",
          messages: [userMsg],
        };
        this.conversations.update(list => [newConv, ...list]);
        this.activeConvId.set(newConvId);
        currentConvId = newConvId;

        // Create conversation in backend
        try {
          const bc = await firstValueFrom(
            this.convService.create(newConv.title, user.id)
          );
          // Update local ID with backend ID
          this.conversations.update(list =>
            list.map(c =>
              c.id === newConvId
                ? { ...c, id: String(bc.idConversation) }
                : c
            )
          );
          this.activeConvId.set(String(bc.idConversation));
          currentConvId = String(bc.idConversation);
        } catch {
          // Keep local if backend fails
        }
      } else {
        this.addMessageToConv(currentConvId, userMsg);
      }
    }

    if (!currentConvId) return;

    this.isTyping.set(true);

    const formType = this.detectFormType(text);
    const detectedType: MessageType = formType ? 'form' : this.detectType(text);

    let aiText = '';
    let fields: FormField[] | undefined;
    let aiSource: string | undefined;
    let aiSources: SourceInfo[] = [];

    if (detectedType === 'form' && formType) {
      aiText = "J'ai détecté que vous souhaitez effectuer une démarche administrative. Veuillez remplir le formulaire ci-dessous pour que je puisse générer votre document :";
      fields = getFormFields(formType);
      aiSource = getSource(detectedType);
      this.appendAiMessage(currentConvId, aiText, detectedType, formType, fields, aiSource);
    } else if (detectedType === 'resume') {
      aiText = "Voici un résumé basé sur votre demande. Ce document aborde les points clés suivants :\n\n1. Introduction au concept principal\n2. Développement des idées et arguments\n3. Conclusion et perspectives d'application";
      aiSource = getSource(detectedType);
      this.appendAiMessage(currentConvId, aiText, detectedType, undefined, undefined, aiSource);
    } else {
      // Use backend RAG chat
      try {
        const convIdNum = Number(currentConvId);
        if (convIdNum) {
          const token = localStorage.getItem('access_token');
          
          const aiMsgId = 'msg-' + Date.now() + '-ai';
          const aiMsg: Message = {
            id: aiMsgId,
            text: '',
            sender: 'ai',
            timestamp: new Date(),
            type: detectedType,
          };
          this.addMessageToConv(currentConvId, aiMsg);

          const response = await fetch(`${this.API}/chat/stream`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
              conversation_id: convIdNum,
              question: text.trim(),
              use_graph: true,
              top_k: 30,
            })
          });

          if (!response.ok || !response.body) {
            throw new Error('Network error');
          }

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let aiTextStream = '';
          let aiSources: SourceInfo[] = [];
          let aiSourceText: string | undefined;
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.substring(6));
                  if (data.type === 'sources') {
                    const rawSources = data.sources || [];
                    aiSources = rawSources.slice(0, 5).map((s: any) => ({
                      id: s.id, site: s.site, title: s.title,
                      section: s.section, type: s.type, url: s.url
                    }));
                    aiSourceText = rawSources.length > 0
                      ? rawSources.slice(0, 5).map((s: any) => `📄 ${s.title || s.titre || s.id}`).join('\n')
                      : undefined;
                    
                    this.conversations.update(list => list.map(c => c.id === currentConvId ? {
                      ...c, messages: c.messages.map(m => m.id === aiMsgId ? { ...m, sources: aiSources, source: aiSourceText } : m)
                    } : c));
                  } else if (data.type === 'chunk') {
                    aiTextStream += data.text;
                    this.conversations.update(list => list.map(c => c.id === currentConvId ? {
                      ...c, messages: c.messages.map(m => m.id === aiMsgId ? { ...m, text: aiTextStream } : m)
                    } : c));
                  } else if (data.type === 'done') {
                    this.conversations.update(list => list.map(c => c.id === currentConvId ? {
                      ...c, messages: c.messages.map(m => m.id === aiMsgId ? { ...m, suggestions: getSuggestions(detectedType) } : m)
                    } : c));
                  } else if (data.type === 'error') {
                    aiTextStream += `\nErreur: ${data.detail}`;
                    this.conversations.update(list => list.map(c => c.id === currentConvId ? {
                      ...c, messages: c.messages.map(m => m.id === aiMsgId ? { ...m, text: aiTextStream } : m)
                    } : c));
                  }
                } catch (e) {}
              }
            }
          }
          this.isTyping.set(false);
          return;
        } else {
          aiText = 'Je comprends votre demande. Comment puis-je vous aider ?';
          aiSource = getSource(detectedType);
          aiSources = [];
        }
      } catch (err) {
        aiText = 'Désolé, je n\'ai pas pu traiter votre demande pour le moment. Veuillez réessayer.';
        aiSource = undefined;
        aiSources = [];
      }
      this.appendAiMessage(currentConvId, aiText, detectedType, undefined, undefined, aiSource, aiSources);
    }
  }

  private appendAiMessage(
    convId: string,
    text: string,
    type: MessageType,
    formType?: FormType,
    formFields?: FormField[],
    source?: string,
    sources?: SourceInfo[],
  ): void {
    const aiMsg: Message = {
      id: 'msg-' + Date.now() + '-ai',
      text,
      sender: 'ai',
      timestamp: new Date(),
      type,
      source,
      sources,
      suggestions: type !== 'form' ? getSuggestions(type) : undefined,
      formType,
      formFields,
      formData: formFields ? {} : undefined,
    };
    this.addMessageToConv(convId, aiMsg);
    this.isTyping.set(false);
  }

  async submitEdit(msgId: string, text: string): Promise<void> {
    const activeId = this.activeConvId();
    if (!activeId || !text.trim()) return;

    this.conversations.update(list =>
      list.map(c => {
        if (c.id !== activeId) return c;
        const msgIndex = c.messages.findIndex(m => m.id === msgId);
        if (msgIndex === -1) return c;
        const updatedMessage = { ...c.messages[msgIndex], text: text.trim() };
        const updatedMessages = [...c.messages.slice(0, msgIndex), updatedMessage];
        return { ...c, messages: updatedMessages };
      })
    );

    this.isTyping.set(true);
    await this.sendMessage(text, true);
  }

  async regenerateMessage(msgId: string): Promise<void> {
    const activeId = this.activeConvId();
    if (!activeId) return;

    let userText = '';

    this.conversations.update(list =>
      list.map(c => {
        if (c.id !== activeId) return c;
        const msgIndex = c.messages.findIndex(m => m.id === msgId);
        if (msgIndex === -1) return c;

        for (let i = msgIndex - 1; i >= 0; i--) {
          if (c.messages[i].sender === 'user') {
            userText = c.messages[i].text;
            break;
          }
        }
        if (!userText) userText = 'Aide';

        const updatedMessages = c.messages.filter(m => m.id !== msgId);
        return { ...c, messages: updatedMessages };
      })
    );

    if (userText) {
      this.isTyping.set(true);
      await this.sendMessage(userText, true);
    }
  }

  private addMessageToConv(convId: string, msg: Message): void {
    this.conversations.update(list =>
      list.map(c => c.id === convId ? { ...c, messages: [...c.messages, msg] } : c)
    );
  }

  updateFormField(msgId: string, fieldName: string, value: string): void {
    this.conversations.update(list =>
      list.map(c =>
        c.id === this.activeConvId()
          ? {
              ...c,
              messages: c.messages.map(m =>
                m.id === msgId
                  ? { ...m, formData: { ...(m.formData || {}), [fieldName]: value } }
                  : m
              ),
            }
          : c
      )
    );
  }

  getFormData(msgId: string): Record<string, string> {
    return this.messages().find(m => m.id === msgId)?.formData || {};
  }

  async submitForm(msg: Message): Promise<void> {
    if (!msg.formType) return;

    this.conversations.update(list =>
      list.map(c =>
        c.id === this.activeConvId()
          ? { ...c, messages: c.messages.map(m => m.id === msg.id ? { ...m, formSubmitted: true } : m) }
          : c
      )
    );

    this.isTyping.set(true);
    const data = this.getFormData(msg.id);
    const generatedText = generateDocument(msg.formType, data);

    const aiMsg: Message = {
      id: 'msg-' + Date.now() + '-gen',
      text: generatedText,
      sender: 'ai',
      timestamp: new Date(),
      type: 'normal',
      source: 'Générateur de documents UniHelp',
      suggestions: ['Générer un autre document', 'Question administrative'],
    };

    const activeId = this.activeConvId();
    if (activeId) {
      this.addMessageToConv(activeId, aiMsg);
    }
    this.isTyping.set(false);
  }
}
