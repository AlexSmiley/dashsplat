// Icons drawn for this app in the visual language of the Tesla UI (white, rounded, 24-unit grid).
const svg = (body: string, cls = "") => `<svg class="ic ${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;

export const icons = {
  // bottom left
  map: svg('<path d="M12 15.2s-4.6-4-4.6-7.6a4.6 4.6 0 0 1 9.2 0c0 3.6-4.6 7.6-4.6 7.6z" fill="currentColor" stroke="none"/><circle cx="12" cy="7.6" r="1.7" fill="#1b1c1f" stroke="none"/><path d="M8 14.6 3.5 17.6 12 21.5l8.5-3.9L16 14.6"/>'),
  moon: svg('<path d="M19.5 14.6A8.2 8.2 0 0 1 9.4 4.5a8.2 8.2 0 1 0 10.1 10.1z" fill="currentColor" stroke="none"/>'),
  // media-card style transport
  prev: svg('<path d="M7 5.5v13" stroke-width="2.6"/><path d="M18.5 5.8v12.4L9.4 12z" fill="currentColor" stroke="none"/>'),
  next: svg('<path d="M17 5.5v13" stroke-width="2.6"/><path d="M5.5 5.8v12.4L14.6 12z" fill="currentColor" stroke="none"/>'),
  play: svg('<path d="M7.5 5.2v13.6L18.6 12z" fill="currentColor" stroke="none"/>'),
  pause: svg('<rect x="6.6" y="5.4" width="3.6" height="13.2" rx="0.8" fill="currentColor" stroke="none"/><rect x="13.8" y="5.4" width="3.6" height="13.2" rx="0.8" fill="currentColor" stroke="none"/>'),
  memory: svg('<path d="M3.8 12a8.2 8.2 0 1 0 2.4-5.8"/><path d="M3.6 3.8v3.6h3.6"/><path d="M12 7.6V12l3 2"/>'),
  camera: svg('<path d="M3.5 8.2h3.4L8.6 5.8h6.8l1.7 2.4h3.4v10.3h-17z"/><circle cx="12" cy="13.1" r="3.2"/>'),
  // trip card
  pinPlus: svg('<path d="M10.5 20s-5.2-4.6-5.2-8.6a5.2 5.2 0 0 1 10.4 0c0 4-5.2 8.6-5.2 8.6z" fill="currentColor" stroke="none"/><circle cx="10.5" cy="11.3" r="1.9" fill="#1b1c1f" stroke="none"/><path d="M18.6 4.4v5M16.1 6.9h5" stroke-width="1.8"/>'),
  dots: svg('<circle cx="5.5" cy="12" r="1.5" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none"/><circle cx="18.5" cy="12" r="1.5" fill="currentColor" stroke="none"/>'),
  // status
  lock: svg('<rect x="6" y="10.4" width="12" height="9.6" rx="1.8" fill="currentColor" stroke="none"/><path d="M8.6 10.4V8a3.4 3.4 0 0 1 6.8 0v2.4" stroke-width="2.2"/>'),
  cloud: svg('<path d="M7.2 18.2h10.2a3.9 3.9 0 0 0 .4-7.8 5.3 5.3 0 0 0-10.2 1.3 3.3 3.3 0 0 0-.4 6.5z" fill="currentColor" stroke="none"/>'),
  airbag: svg('<circle cx="12" cy="12" r="9.2" stroke-width="1.7"/><circle cx="10.4" cy="7.7" r="1.3" fill="currentColor" stroke="none"/><path d="M9.6 10.3l.9 4.4h4l1.6 3.1M8.8 10.7l-.8 5.3h3.6" stroke-width="1.6"/><path d="M5.6 18.4 18.4 5.6" stroke-width="1.7"/>'),
  lowBeam: svg('<path d="M12.5 5.5c4.6 0 7.5 2.9 7.5 6.5s-2.9 6.5-7.5 6.5z" stroke-width="2"/><path d="M3.5 7.5h6M3.5 10.5h6M3.5 13.5h6M3.5 16.5h6" stroke-width="2"/>'),
  parkLights: svg('<path d="M14 5.5c3.9 0 6.5 2.9 6.5 6.5s-2.6 6.5-6.5 6.5z" stroke-width="2"/><path d="M3.5 8.5h7M3.5 15.5h7M7 6v12" stroke-width="2"/>'),
  chevronDown: svg('<path d="M7 10l5 5 5-5"/>', "chev"),
  // next turn
  left: svg('<path d="M16 20.5v-7.3a3.2 3.2 0 0 0-3.2-3.2H5.5M9.5 6l-4 4 4 4" stroke-width="2.4"/>', "turn"),
  right: svg('<path d="M8 20.5v-7.3a3.2 3.2 0 0 1 3.2-3.2h7.3M14.5 6l4 4-4 4" stroke-width="2.4"/>', "turn"),
  arrive: svg('<circle cx="12" cy="12" r="5.4" stroke-width="2.4"/><circle cx="12" cy="12" r="1.8" fill="currentColor" stroke="none"/><path d="M2.5 12h2.8" stroke-width="2.4"/><path d="M4.8 9.6 2.4 12l2.4 2.4" stroke-width="2"/>', "turn"),
};
