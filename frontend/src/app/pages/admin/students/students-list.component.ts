import { Component, signal, computed, inject, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { TranslateModule } from '@ngx-translate/core';
import { HttpClient } from '@angular/common/http';
import { animate, style, transition, trigger, state } from '@angular/animations';
import { environment } from '../../../../environments/environment';

interface Student {
  id: number;
  name: string;
  email: string;
  phone: string;
  birthDate: string;
  cin: string;
  passport: string;
  address: string;
  country: string;
  uni: string;
  field: string;
  level: string;
  courseTime: string;
  date: string;
  academicYear: string;
  lastDiploma: string;
  diplomaYear: string;
  knownBy: string;
  status: 'Actif' | 'Inactif' | 'En attente';
  studentId: string;
  photo?: string | null;
}

@Component({
  selector: 'app-students-list',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule, TranslateModule],
  templateUrl: './students-list.component.html',
  animations: [
    trigger('fadeIn', [
      transition(':enter', [style({ opacity: 0 }), animate('200ms ease-out', style({ opacity: 1 }))]),
      transition(':leave', [animate('200ms ease-in', style({ opacity: 0 }))])
    ]),
    trigger('modalSlide', [
      transition(':enter', [
        style({ transform: 'scale(0.95) translateY(20px)', opacity: 0 }),
        animate('300ms cubic-bezier(0.34, 1.56, 0.64, 1)', style({ transform: 'scale(1) translateY(0)', opacity: 1 }))
      ]),
      transition(':leave', [
        animate('200ms ease-in', style({ transform: 'scale(0.95) translateY(20px)', opacity: 0 }))
      ])
    ])
  ]
})
export class StudentsListComponent implements OnInit {
  private http = inject(HttpClient);
  private readonly API = environment.apiUrl;

  searchQuery = signal('');
  uniFilter = signal('');
  fieldFilter = signal('');
  statusFilter = signal('');
  isFilterOpen = signal(false);
  selectedStudent = signal<Student | null>(null);
  currentPage = signal(1);
  itemsPerPage = 5;
  loading = signal(true);

  students = signal<Student[]>([]);

  ngOnInit(): void {
    this.loadStudents();
  }

  private loadStudents(): void {
    this.http.get<any[]>(`${this.API}/etudiants`).subscribe({
      next: (list) => {
        this.students.set(
          list.map((e) => ({
            id: e.id,
            name: `${e.prenom} ${e.nom}`,
            email: e.email,
            phone: e.telephone || '-',
            birthDate: e.dateNaissance || '-',
            cin: e.CIN || '-',
            passport: e.passport || '-',
            address: e.adresse || '-',
            country: e.pays || '-',
            uni: e.universite || '-',
            field: e.diplomeSouhaite || '-',
            level: e.niveau || '-',
            courseTime: e.cours || '-',
            date: '-',
            academicYear: '-',
            lastDiploma: e.dernierDiplomeObtenu || '-',
            diplomaYear: e.anneeObtention || '-',
            knownBy: e.sourceConnaissanceUniversite || '-',
            status: (e.estActif ?? true) ? 'Actif' as const : 'Inactif' as const,
            studentId: String(e.id),
            photo: e.photo
          }))
        );
        this.loading.set(false);
      },
      error: () => this.loading.set(false),
    });
  }
  fields = computed(() => {
    const set = new Set(this.students().map(s => s.field).filter(Boolean));
    return Array.from(set).sort();
  });

  filteredStudents = computed(() => {
    const q = this.searchQuery().toLowerCase();
    const uni = this.uniFilter();
    const field = this.fieldFilter();
    const status = this.statusFilter();

    return this.students().filter(s => {
      const matchesSearch = s.name.toLowerCase().includes(q) || s.email.toLowerCase().includes(q) || s.studentId.toLowerCase().includes(q);
      const matchesUni = !uni || s.uni === uni;
      const matchesField = !field || s.field === field;
      const matchesStatus = !status || s.status.toLowerCase() === status.toLowerCase();
      return matchesSearch && matchesUni && matchesField && matchesStatus;
    });
  });

  filteredCount = computed(() => this.filteredStudents().length);
  totalPages = computed(() => Math.ceil(this.filteredCount() / this.itemsPerPage));
  startIndex = computed(() => (this.currentPage() - 1) * this.itemsPerPage);
  endIndex = computed(() => Math.min(this.startIndex() + this.itemsPerPage, this.filteredCount()));
  
  paginatedStudents = computed(() => {
    const start = this.startIndex();
    return this.filteredStudents().slice(start, start + this.itemsPerPage);
  });

  pageNumbers = computed(() => {
    const total = this.totalPages();
    const current = this.currentPage();
    let start = Math.max(1, current - 2);
    let end = Math.min(total, start + 4);
    if (end - start < 4) {
      start = Math.max(1, end - 4);
    }
    return Array.from({ length: Math.max(0, end - start + 1) }, (_, i) => start + i);
  });

  toggleFilter() { this.isFilterOpen.update(v => !v); }
  
  clearFilters() {
    this.uniFilter.set('');
    this.fieldFilter.set('');
    this.statusFilter.set('');
    this.isFilterOpen.set(false);
    this.currentPage.set(1);
  }

  prevPage() { if (this.currentPage() > 1) this.currentPage.update(v => v - 1); }
  nextPage() { if (this.currentPage() < this.totalPages()) this.currentPage.update(v => v + 1); }
  goToPage(p: number) { this.currentPage.set(p); }
  
  viewProfile(student: Student) {
    this.selectedStudent.set(student);
  }

  toggleActive(student: Student) {
    this.http.patch(`${this.API}/etudiants/${student.id}/toggle-active`, {}).subscribe({
      next: (res: any) => {
        student.status = res.estActif ? 'Actif' : 'Inactif';
        this.students.update(st => [...st]);
      }
    });
  }
}
