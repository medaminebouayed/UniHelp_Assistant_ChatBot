import { ApplicationConfig, provideZoneChangeDetection, importProvidersFrom } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideAnimations } from '@angular/platform-browser/animations';
import { provideHttpClient, withFetch, withInterceptors } from '@angular/common/http';
import { HttpClient } from '@angular/common/http';
import { TranslateModule, TranslateLoader } from '@ngx-translate/core';
import { TranslateHttpLoader, provideTranslateHttpLoader } from '@ngx-translate/http-loader';
import { 
  LucideAngularModule, 
  Mail, Lock, Loader2, GraduationCap, LayoutDashboard, MessageSquare, 
  User, MailCheck, HelpCircle, Settings, FileText, Users, Shield, 
  LogOut, ChevronRight, Search, Bell, Menu, X, Plus, Filter, 
  Download, Trash2, Edit, Save, Check, AlertCircle, CheckCircle, XCircle,
  Edit2, Trash, ChevronDown, UploadCloud, Eye, Building, Clock, 
  Sun, Moon, Globe, Info, Palette, Camera, ChevronLeft, Send, Sparkles, Copy, RefreshCw, Paperclip,
  Contact, MapPin, Calendar, BookOpen, Phone, Layers, Megaphone, Star, Radio,
  MoreVertical, MoreHorizontal, Volume2, File, Coins, EyeOff, Folder, UserMinus, UserCheck
} from 'lucide-angular';

import { routes } from './app.routes';
import { authInterceptor } from './interceptors/auth.interceptor';

export const appConfig: ApplicationConfig = {
  providers: [
    provideZoneChangeDetection({ eventCoalescing: true }), 
    provideRouter(routes),
    provideAnimations(),
    provideHttpClient(withFetch(), withInterceptors([authInterceptor])),
    provideTranslateHttpLoader({ prefix: './assets/i18n/', suffix: '.json' }),
    importProvidersFrom(
      TranslateModule.forRoot({
        defaultLanguage: 'fr',
        loader: { provide: TranslateLoader, useClass: TranslateHttpLoader }
      }),
      LucideAngularModule.pick({ 
      Mail, Lock, Loader2, GraduationCap, LayoutDashboard, MessageSquare, 
      User, MailCheck, HelpCircle, Settings, FileText, Users, Shield, 
      LogOut, ChevronRight, Search, Bell, Menu, X, Plus, Filter, 
      Download, Trash2, Edit, Save, Check, AlertCircle, CheckCircle, XCircle,
      Edit2, Trash, ChevronDown, UploadCloud, Eye, Building, Clock, 
      Sun, Moon, Globe, Info, Palette, Camera, ChevronLeft, Send, Sparkles, Copy, RefreshCw, Paperclip,
      Contact, MapPin, Calendar, BookOpen, Phone, Layers, Megaphone, Star, Radio,
      MoreVertical, MoreHorizontal, Volume2, File, Coins, EyeOff, Folder, UserMinus, UserCheck
    }))
  ]
};
