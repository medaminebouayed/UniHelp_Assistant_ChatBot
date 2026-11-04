import { Component, signal, inject, OnInit, AfterViewInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterModule } from '@angular/router';
import { LucideAngularModule } from 'lucide-angular';
import { TranslateModule } from '@ngx-translate/core';
import { AuthService } from '../../../services/auth.service';
import { DashboardService } from '../../../services/dashboard.service';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

@Component({
  selector: 'app-student-dashboard',
  standalone: true,
  imports: [CommonModule, RouterModule, LucideAngularModule, TranslateModule],
  templateUrl: './dashboard.component.html'
})
export class DashboardComponent implements OnInit, AfterViewInit, OnDestroy {
  private authService = inject(AuthService);
  private dashboardService = inject(DashboardService);
  private router = inject(Router);

  user = this.authService.user;
  loading = signal(true);
  chartTab = signal<'semaine' | 'mois'>('semaine');
  private chart: Chart | null = null;

  stats = signal<{
    conversations: number;
    emails: number;
    documents: number;
    favorites: number;
  }>({ conversations: 0, emails: 0, documents: 0, favorites: 0 });

  recentActivity = signal<{ id: number; action: string; time: string; type: string }[]>([]);
  favoriteMessages = signal<{ id: number; text: string; source: string; time: string }[]>([]);

  weeklyData = [
    { day: 'Lun', messages: 0, activite: 0 },
    { day: 'Mar', messages: 0, activite: 0 },
    { day: 'Mer', messages: 0, activite: 0 },
    { day: 'Jeu', messages: 0, activite: 0 },
    { day: 'Ven', messages: 0, activite: 0 },
    { day: 'Sam', messages: 0, activite: 0 },
    { day: 'Dim', messages: 0, activite: 0 },
  ];

  monthlyData = [
    { day: 'Jan', messages: 0, activite: 0 },
    { day: 'Fév', messages: 0, activite: 0 },
    { day: 'Mar', messages: 0, activite: 0 },
    { day: 'Avr', messages: 0, activite: 0 },
    { day: 'Mai', messages: 0, activite: 0 },
    { day: 'Juin', messages: 0, activite: 0 },
    { day: 'Juil', messages: 0, activite: 0 },
    { day: 'Août', messages: 0, activite: 0 },
    { day: 'Sep', messages: 0, activite: 0 },
    { day: 'Oct', messages: 0, activite: 0 },
    { day: 'Nov', messages: 0, activite: 0 },
    { day: 'Déc', messages: 0, activite: 0 },
  ];

  activityData = this.weeklyData;
  progressionData = [
    { name: 'Conversations', value: 0 },
    { name: 'Emails', value: 0 },
    { name: 'Documents', value: 0 },
  ];

  progressionTab = signal<'bar' | 'pie'>('pie');
  private progChart: Chart | null = null;
  private PROGRESSION_COLORS = ['#1d4b99', '#8db2edff', '#bdd9f4ff'];

  ngOnInit(): void {
    this.loadDashboard();
  }

  private loadDashboard(): void {
    this.dashboardService.getStats().subscribe({
      next: (res: any) => {
        if (res.role === 'etudiant') {
          this.stats.set({
            conversations: res.total_conversations ?? 0,
            emails: res.total_emails ?? 0,
            documents: res.total_documents ?? 0,
            favorites: res.total_favorites ?? 0,
          });
          this.recentActivity.set(res.recent_activity ?? []);
          this.favoriteMessages.set(res.favorite_conversations ?? []);
          this.progressionData = [
            { name: 'Conversations', value: res.total_conversations ?? 0 },
            { name: 'Emails', value: res.total_emails ?? 0 },
            { name: 'Documents', value: res.total_documents ?? 0 },
          ];
          if (res.conversations_this_week > 0) {
            this.weeklyData = this.weeklyData.map((d, i) => ({
              ...d,
              activite: i < res.conversations_this_week ? Math.min(res.conversations_this_week, 5) : 0,
            }));
          }
          if (res.conversations_this_month > 0) {
            const perMonth = Math.max(1, Math.round(res.conversations_this_month / 12));
            this.monthlyData = this.monthlyData.map(d => ({
              ...d,
              activite: perMonth + Math.floor(Math.random() * perMonth),
            }));
          }
        }
        this.loading.set(false);
      },
      error: () => this.loading.set(false),
    });
  }

  ngAfterViewInit() {
    setTimeout(() => {
      this.renderChart();
      this.renderProgressionChart();
    }, 100);
  }

  ngOnDestroy() {
    if (this.chart) { this.chart.destroy(); }
    if (this.progChart) { this.progChart.destroy(); }
  }

  activeData() {
    return this.chartTab() === 'semaine' ? this.weeklyData : this.monthlyData;
  }

  renderChart() {
    const canvas = document.getElementById('activityChart') as HTMLCanvasElement;
    if (!canvas) return;
    if (this.chart) { this.chart.destroy(); }
    const data = this.activeData();
    this.chart = new Chart(canvas, {
      type: 'line',
      data: {
        labels: data.map(d => d.day),
        datasets: [{
          label: 'Activité',
          data: data.map(d => d.activite),
          borderColor: '#1d4b99',
          backgroundColor: 'rgba(29, 75, 153, 0.1)',
          borderWidth: 3,
          fill: true,
          tension: 0.4,
          pointBackgroundColor: '#1d4b99',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          pointRadius: 5,
          pointHoverRadius: 7
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#ffffff',
            titleColor: '#64748b',
            bodyColor: '#1e293b',
            borderColor: '#e2e8f0',
            borderWidth: 1,
            padding: 12,
            cornerRadius: 12,
            callbacks: {
              label: (ctx: any) => ` ${ctx.parsed.y} action(s)`
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { size: 12 } }
          },
          y: {
            beginAtZero: true,
            grid: { color: '#f1f5f9' },
            ticks: {
              color: '#94a3b8',
              font: { size: 12 },
              maxTicksLimit: 6,
              callback: (v: any) => Number.isInteger(v) ? v : ''
            }
          }
        }
      }
    });
  }

  setChartTab(tab: 'semaine' | 'mois') {
    this.chartTab.set(tab);
    setTimeout(() => this.renderChart(), 0);
  }

  setProgressionTab(tab: 'bar' | 'pie') {
    this.progressionTab.set(tab);
    setTimeout(() => this.renderProgressionChart(), 0);
  }

  renderProgressionChart() {
    const canvas = document.getElementById('progressionChart') as HTMLCanvasElement;
    if (!canvas) return;
    if (this.progChart) { this.progChart.destroy(); }
    const type = this.progressionTab() === 'bar' ? 'bar' : 'pie';
    this.progChart = new Chart(canvas, {
      type: type,
      data: {
        labels: this.progressionData.map(d => d.name),
        datasets: [{
          data: this.progressionData.map(d => d.value),
          backgroundColor: this.PROGRESSION_COLORS,
          borderRadius: type === 'bar' ? 8 : 0,
          borderWidth: type === 'pie' ? 2 : 0,
          borderColor: '#ffffff'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: type === 'pie',
            position: 'bottom',
            labels: { usePointStyle: true, padding: 20, font: { size: 12 } }
          },
          tooltip: {
            backgroundColor: '#ffffff',
            titleColor: '#64748b',
            bodyColor: '#1e293b',
            borderColor: '#e2e8f0',
            borderWidth: 1,
            padding: 12,
            cornerRadius: 12
          }
        },
        scales: type === 'bar' ? {
          x: { grid: { display: false }, ticks: { font: { size: 12 } } },
          y: { beginAtZero: true, grid: { color: '#f1f5f9' }, ticks: { display: true, color: '#94a3b8', font: { size: 12 }, stepSize: 35 } }
        } : { x: { display: false }, y: { display: false } }
      }
    });
  }

  firstName() {
    const name = this.user()?.name;
    return name ? name.split(' ')[0] : 'Étudiant';
  }

  navigateToChat() {
    this.router.navigate(['/student/chatbot']);
  }

  getActivityIcon(type: string): string {
    switch (type) {
      case 'chat': return 'message-square';
      case 'email': return 'mail';
      case 'doc': return 'file-text';
      case 'profile': return 'user';
      default: return 'help-circle';
    }
  }

  getActivityColor(type: string): string {
    switch (type) {
      case 'chat': return 'text-[#1d4b99]';
      case 'email': return 'text-purple-500';
      case 'doc': return 'text-emerald-500';
      case 'profile': return 'text-slate-400';
      default: return 'text-slate-400';
    }
  }
}
