import { defineThemeConfig } from 'vuepress-theme-plume'
import { navbar } from './navbar'
import { notes } from './notes'

/**
 * @see https://theme-plume.vuejs.press/config/basic/
 */
export default defineThemeConfig({
  logo: 'https://theme-plume.vuejs.press/plume.png',
  // your git repo url
  docsRepo: 'https://github.com/Velvet0314/Velvet-Blog',
  docsDir: 'docs',
  appearance: true,
  footer: { copyright: 'Copyright © 2024-present Velvet' },
  // transition: {appearance: false},

  profile: {
    avatar: 'https://image.velvet-notes.org/blog/avatar.png',
    name: 'Velvet',
    description: '血潮如铁，心似琉璃',
    // circle: true,
    // location: '',
    // organization: '',
  },

  navbar,
  notes,
  social: [
    { icon: 'github', link: 'https://github.com/Velvet0314' },
  ],

})