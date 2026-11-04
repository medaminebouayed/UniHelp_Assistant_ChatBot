import { Component, signal, inject, OnInit, AfterViewInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LucideAngularModule } from 'lucide-angular';
import { animate, style, transition, trigger } from '@angular/animations';
import { RouterLink } from '@angular/router';
import { TranslateModule } from '@ngx-translate/core';
import { Chart, registerables } from 'chart.js';
import { DashboardService } from '../../../services/dashboard.service';
import { AuthService } from '../../../services/auth.service';

Chart.register(...registerables);

interface Stat {
  label: string;
  value: string;
  icon: string;
  color: string;
  bg: string;
  fill?: boolean;
}

interface ChartData {
  name: string;
  inscrits: number;
}

@Component({
  selector: 'app-admin-dashboard',
  standalone: true,
  imports: [CommonModule, LucideAngularModule, RouterLink, TranslateModule],
  templateUrl: './admin-dashboard.component.html',
  animations: [
    trigger('fadeInUp', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(10px)' }),
        animate('300ms ease-out', style({ opacity: 1, transform: 'translateY(0)' }))
      ])
    ])
  ]
})
export class AdminDashboardComponent implements OnInit, AfterViewInit, OnDestroy {
  private dashboardService = inject(DashboardService);
  private authService = inject(AuthService);

  chartTab = signal<'semaine' | 'mois'>('semaine');
  private chart: Chart | null = null;
  loading = signal(true);

  stats: Stat[] = [
    { label: 'admin.totalStudents', value: '0', icon: 'users', color: 'text-emerald-600', bg: 'bg-emerald-50' },
    { label: 'admin.totalDocuments', value: '0', icon: 'file-text', color: 'text-blue-600', bg: 'bg-blue-50' },
    { label: 'admin.totalEmails', value: '0', icon: 'star', color: 'text-amber-500', bg: 'bg-amber-50', fill: true },
    { label: 'admin.totalAdmins', value: '0', icon: 'shield', color: 'text-purple-600', bg: 'bg-purple-50' }
  ];

  ngOnInit(): void {
    this.loadStats();
  }

  private loadStats(): void {
    this.dashboardService.getStats().subscribe({
      next: (res: any) => {
        if (res.role === 'admin') {
          this.stats = [
            { label: 'admin.totalStudents', value: String(res.total_etudiants ?? 0), icon: 'users', color: 'text-emerald-600', bg: 'bg-emerald-50' },
            { label: 'admin.totalDocuments', value: String(res.total_documents ?? 0), icon: 'file-text', color: 'text-blue-600', bg: 'bg-blue-50' },
            { label: 'admin.totalEmails', value: String(res.total_emails ?? 0), icon: 'star', color: 'text-amber-500', bg: 'bg-amber-50', fill: true },
            { label: 'admin.totalAdmins', value: String(res.total_admins ?? 0), icon: 'shield', color: 'text-purple-600', bg: 'bg-purple-50' }
          ];
          this.recentStudents = res.recent_students ?? [];
          if (res.weekly_data) this.weeklyData = res.weekly_data;
          if (res.monthly_data) this.monthlyData = res.monthly_data;
          if (res.top_questions) this.topQuestions = res.top_questions;
          if (res.categories) this.categories = res.categories;
          this.renderChart();
        }
        this.loading.set(false);
      },
      error: () => this.loading.set(false),
    });
  }

  weeklyData: ChartData[] = [
    { name: 'Lun', inscrits: 12 }, { name: 'Mar', inscrits: 18 }, { name: 'Mer', inscrits: 15 },
    { name: 'Jeu', inscrits: 25 }, { name: 'Ven', inscrits: 30 }, { name: 'Sam', inscrits: 10 }, { name: 'Dim', inscrits: 5 }
  ];

  monthlyData: ChartData[] = [
    { name: 'Jan', inscrits: 40 }, { name: 'Fév', inscrits: 30 }, { name: 'Mar', inscrits: 45 },
    { name: 'Avr', inscrits: 50 }, { name: 'Mai', inscrits: 65 }, { name: 'Juin', inscrits: 85 },
    { name: 'Juil', inscrits: 90 }, { name: 'Août', inscrits: 60 }, { name: 'Sep', inscrits: 110 },
    { name: 'Oct', inscrits: 125 }, { name: 'Nov', inscrits: 115 }, { name: 'Déc', inscrits: 135 }
  ];

  recentStudents: { id: number; name: string; field: string; status: string }[] = [];

  topQuestions: any[] = [];
  categories: any[] = [];

  activeData() {
    return this.chartTab() === 'semaine' ? this.weeklyData : this.monthlyData;
  }

  ngAfterViewInit() {
    this.renderChart();
  }

  ngOnDestroy() {
    if (this.chart) {
      this.chart.destroy();
    }
  }

  renderChart() {
    const canvas = document.getElementById('inscriptionsChart') as HTMLCanvasElement;
    if (!canvas) return;

    if (this.chart) {
      this.chart.destroy();
    }

    const data = this.activeData();

    this.chart = new Chart(canvas, {
      type: 'line',
      data: {
        labels: data.map(d => d.name),
        datasets: [{
          label: 'Inscriptions',
          data: data.map(d => d.inscrits),
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
              label: (ctx: any) => ` ${ctx.parsed.y} inscription(s)`
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
              stepSize: 25,
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
}
