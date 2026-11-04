import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LucideAngularModule } from 'lucide-angular';
import { TranslateModule } from '@ngx-translate/core';

@Component({
  selector: 'app-uni-help-logo',
  standalone: true,
  imports: [CommonModule, LucideAngularModule, TranslateModule],
  templateUrl: './uni-help-logo.component.html'
})
export class UniHelpLogoComponent {
  @Input() size: 'small' | 'sm' | 'medium' | 'md' | 'large' | 'lg' | 'xlarge' | 'xl' = 'medium';
  @Input() color: string = '#2563eb';
  @Input() iconName: string = 'graduation-cap';
  @Input() showText: boolean = true;

  get sizeClasses(): string {
    switch (this.size) {
      case 'sm':
      case 'small': return 'w-8 h-8 rounded-xl';
      case 'lg':
      case 'large': return 'w-16 h-16 rounded-2xl';
      case 'xl':
      case 'xlarge': return 'w-18 h-18 rounded-2xl';
      case 'md':
      case 'medium':
      default: return 'w-12 h-12 rounded-2xl';
    }
  }

  get iconSizeClasses(): string {
    switch (this.size) {
      case 'sm':
      case 'small': return 'w-4 h-4';
      case 'lg':
      case 'large': return 'w-10 h-10';
      case 'xl':
      case 'xlarge': return 'w-12 h-12';
      case 'md':
      case 'medium':
      default: return 'w-6 h-6';
    }
  }

  get textSizeClasses(): string {
    switch (this.size) {
      case 'sm':
      case 'small': return 'text-lg';
      case 'lg':
      case 'large': return 'text-4xl';
      case 'xl':
      case 'xlarge': return 'text-5xl';
      case 'md':
      case 'medium':
      default: return 'text-2xl';
    }
  }
}
