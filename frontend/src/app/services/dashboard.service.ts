import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export interface StudentStats {
  user_id: number;
  role: string;
  total_conversations: number;
  total_emails: number;
  total_documents: number;
  recent_activity: any[];
}

export interface AdminStats {
  user_id: number;
  role: string;
  total_etudiants: number;
  total_documents: number;
  total_emails: number;
  total_admins: number;
  total_conversations: number;
}

@Injectable({ providedIn: 'root' })
export class DashboardService {
  private readonly API = environment.apiUrl;

  constructor(private http: HttpClient) {}

  getStats(): Observable<StudentStats | AdminStats> {
    return this.http.get<StudentStats | AdminStats>(`${this.API}/dashboard/stats`);
  }
}
