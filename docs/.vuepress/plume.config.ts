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
    avatar: 'https://s21.ax1x.com/2024/11/10/pA629aj.jpg',
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