import { Component, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';

@Component({
  selector: 'app-auth-layout',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './auth-layout.component.html',
})
export class AuthLayoutComponent implements AfterViewInit {
  @ViewChild('bgVideo') bgVideo!: ElementRef<HTMLVideoElement>;

  ngAfterViewInit() {
    if (this.bgVideo && this.bgVideo.nativeElement) {
      this.bgVideo.nativeElement.muted = true;
      this.bgVideo.nativeElement.play().catch(e => console.error('Video autoplay prevented:', e));
    }
  }
}
