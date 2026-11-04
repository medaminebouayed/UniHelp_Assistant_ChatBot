import { Component, signal, computed, inject, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { TranslateModule } from '@ngx-translate/core';
import { animate, style, transition, trigger, state } from '@angular/animations';
import { DocumentService, BackendDocument } from '../../../services/document.service';

interface Document {
  id: number;
  title: string;
  category: string;
  date: string;
  size: string;
}

@Component({
  selector: 'app-documents',
  standalone: true,
  imports: [CommonModule, FormsModule, LucideAngularModule, TranslateModule],
  templateUrl: './documents.component.html',
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
export class DocumentsComponent implements OnInit {
  private docService = inject(DocumentService);

  documents = signal<Document[]>([]);
  loading = signal(true);

  ngOnInit(): void {
    this.loadDocuments();
  }

  private loadDocuments(): void {
    this.docService.getAll({ limit: 200 }).subscribe({
      next: (list) => {
        this.documents.set(
          list.map((d: any) => ({
            id: d.idDocument,
            title: d.titre,
            category: d.categorie || 'Non classé',
            date: d.dateAjout ? new Date(d.dateAjout).toLocaleDateString('fr-FR') : '-',
            size: d.taille ? `${d.taille} KB` : '-',
          }))
        );
        this.loading.set(false);
      },
      error: () => this.loading.set(false),
    });
  }

  searchQuery = signal('');
  categoryFilter = signal('');
  isFilterOpen = signal(false);
  isModalOpen = signal(false);
  isDeleteModalOpen = signal(false);
  editingDoc = signal<Document | null>(null);
  currentPage = signal(1);
  itemsPerPage = 5;

  formData = { title: '', category: 'Syllabus' };
  selectedFile: File | null = null;
  categories = ['Syllabus', 'Administration', 'Examens', 'Cours', 'Guide'];

  filteredDocs = computed(() => {
    const q = this.searchQuery().toLowerCase();
    const cat = this.categoryFilter();
    return this.documents().filter(d => {
      const ms = d.title.toLowerCase().includes(q) || d.category.toLowerCase().includes(q);
      const mc = !cat || d.category === cat;
      return ms && mc;
    });
  });

  filteredCount = computed(() => this.filteredDocs().length);
  totalPages = computed(() => Math.ceil(this.filteredCount() / this.itemsPerPage));
  startIndex = computed(() => (this.currentPage() - 1) * this.itemsPerPage);
  endIndex = computed(() => Math.min(this.startIndex() + this.itemsPerPage, this.filteredCount()));

  paginatedDocs = computed(() => {
    const start = this.startIndex();
    return this.filteredDocs().slice(start, start + this.itemsPerPage);
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
    this.categoryFilter.set('');
    this.isFilterOpen.set(false);
    this.currentPage.set(1);
  }

  openAddModal() {
    this.editingDoc.set(null);
    this.formData = { title: '', category: 'Syllabus' };
    this.selectedFile = null;
    this.isModalOpen.set(true);
  }

  openEditModal(doc: Document) {
    this.editingDoc.set(doc);
    this.formData = { title: doc.title, category: doc.category };
    this.selectedFile = null;
    this.isModalOpen.set(true);
  }

  openDeleteModal(doc: Document) {
    this.editingDoc.set(doc);
    this.isDeleteModalOpen.set(true);
  }

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    this.selectedFile = input.files?.length ? input.files[0] : null;
  }

  handleSave() {
    const editing = this.editingDoc();
    const fd = new FormData();
    fd.append('title', this.formData.title);
    fd.append('category', this.formData.category);
    fd.append('site', 'IIT');

    const refresh = () => this.loadDocuments();

    if (editing) {
      if (this.selectedFile) {
        fd.append('file', this.selectedFile);
        this.docService.update(editing.id, fd).subscribe({ next: refresh, error: refresh });
      } else {
        refresh();
      }
    } else {
      if (this.selectedFile) {
        fd.append('file', this.selectedFile);
        this.docService.upload(fd).subscribe({ next: refresh, error: refresh });
      } else {
        this.docService.create({
          titre: this.formData.title,
          categorie: this.formData.category,
          site: 'IIT',
        }).subscribe({ next: refresh, error: refresh });
      }
    }
    this.isModalOpen.set(false);
  }

  handleDelete() {
    const doc = this.editingDoc();
    if (doc) {
      this.docService.delete(doc.id).subscribe({
        next: () => this.loadDocuments(),
        error: () => this.loadDocuments(),
      });
    }
    this.isDeleteModalOpen.set(false);
  }

  prevPage() { if (this.currentPage() > 1) this.currentPage.update(v => v - 1); }
  nextPage() { if (this.currentPage() < this.totalPages()) this.currentPage.update(v => v + 1); }
  goToPage(p: number) { this.currentPage.set(p); }
}
