import { defineNoteConfig, defineNotesConfig } from 'vuepress-theme-plume'

const D2L = defineNoteConfig({
  dir: 'D2L',
  link: '/D2L/',
  sidebar: ['','引言'],
})

const Memo = defineNoteConfig({
  dir: 'Memo',
  link: '/Memo/',
  sidebar: ['','molecule','ssh','git'],
})

export const notes = defineNotesConfig({
  dir: 'notes',
  link: '/',
  notes: [D2L,Memo],
})