import { defineClientConfig } from 'vuepress/client'
// import RepoCard from 'vuepress-theme-plume/features/RepoCard.vue'
// import CustomComponent from './theme/components/Custom.vue'
import './styles/index.css'
// import './theme/styles/custom.css'

export default defineClientConfig({
  enhance({ app }) {
    // app.component('RepoCard', RepoCard)
    // app.component('CustomComponent', CustomComponent)
  },
})
