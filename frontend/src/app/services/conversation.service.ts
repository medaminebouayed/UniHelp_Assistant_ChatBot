import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export interface ConvMessage {
  idMessage: number;
  contenu: string;
  dateEnvoi: string;
  role: string;
  conversation_id: number;
}

export interface BackendConversation {
  idConversation: number;
  titre: string | null;
  estFavori: boolean;
  dateDernierModification: string | null;
  utilisateur_id?: number;
  messages: ConvMessage[];
}

export interface FrontendConversation {
  id: string;
  title: string;
  date: string;
  isFavorite: boolean;
  messages: FrontendMessage[];
}

export interface FrontendMessage {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  type?: string;
  source?: string;
  suggestions?: string[];
}

@Injectable({ providedIn: 'root' })
export class ConversationService {
  private readonly API = environment.apiUrl;

  constructor(private http: HttpClient) {}

  getByUser(userId: number): Observable<BackendConversation[]> {
    return this.http.get<BackendConversation[]>(
      `${this.API}/conversations/utilisateur/${userId}`
    );
  }

  getOne(convId: number): Observable<BackendConversation> {
    return this.http.get<BackendConversation>(
      `${this.API}/conversations/${convId}`
    );
  }

  create(titre: string, utilisateur_id: number): Observable<BackendConversation> {
    return this.http.post<BackendConversation>(`${this.API}/conversations`, {
      titre,
      estFavori: false,
      dateDernierModification: new Date().toISOString().split('T')[0],
      utilisateur_id,
    });
  }

  toggleFavorite(convId: number): Observable<BackendConversation> {
    return this.http.patch<BackendConversation>(
      `${this.API}/conversations/${convId}/favori`,
      {}
    );
  }

  delete(convId: number): Observable<void> {
    return this.http.delete<void>(`${this.API}/conversations/${convId}`);
  }

  mapBackendToFrontend(bc: BackendConversation): FrontendConversation {
    const today = new Date().toISOString().split('T')[0];
    const msgDate = bc.dateDernierModification || today;
    let dateLabel = msgDate;
    if (msgDate === today) dateLabel = "Aujourd'hui";
    else {
      const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];
      if (msgDate === yesterday) dateLabel = 'Hier';
    }

    return {
      id: String(bc.idConversation),
      title: bc.titre || 'Nouvelle conversation',
      date: dateLabel,
      isFavorite: bc.estFavori,
      messages: (bc.messages || []).map((m) => this.mapMessage(m)),
    };
  }

  private mapMessage(m: ConvMessage): FrontendMessage {
    return {
      id: String(m.idMessage),
      text: m.contenu,
      sender: m.role === 'assistant' ? 'ai' : 'user',
      timestamp: new Date(m.dateEnvoi),
    };
  }
}
